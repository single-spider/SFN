from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import physical_to_normalized_action


@dataclass
class ControllerAction:
    normalized: np.ndarray
    physical: np.ndarray


class SFSSController:
    def __init__(
        self,
        gain_xy=0.7,
        gain_yaw=0.7,
        max_xy_mm=2.0,
        max_yaw_deg=2.0,
        confidence_mode="scale",
        confidence_threshold=0.5,
        deadband_xy_mm=0.0,
        deadband_yaw_deg=0.0,
        max_cumulative_xy_mm=float("inf"),
        max_cumulative_yaw_deg=float("inf"),
        max_sign_reversals=0,
    ):
        self.gain_xy = gain_xy
        self.gain_yaw = gain_yaw
        self.max_xy_mm = max_xy_mm
        self.max_yaw_deg = max_yaw_deg
        self.confidence_mode = confidence_mode
        self.confidence_threshold = confidence_threshold
        self.deadband_xy_m = float(deadband_xy_mm) / 1000.0
        self.deadband_yaw_deg = float(deadband_yaw_deg)
        self.max_cumulative_xy_m = float(max_cumulative_xy_mm) / 1000.0
        self.max_cumulative_yaw_deg = float(max_cumulative_yaw_deg)
        self.max_sign_reversals = int(max_sign_reversals)
        if confidence_mode not in {"ignore", "scale", "hold"}:
            raise ValueError("confidence_mode must be ignore, scale, or hold")
        self.reset()

    def reset(self):
        self.cumulative_xy_m = 0.0
        self.cumulative_yaw_deg = 0.0
        self.sign_reversals = 0
        self._last_physical = None
        self.last_hold_reason = None

    def act(self, vsn_output) -> ControllerAction:
        dxy = np.asarray(vsn_output.dxy_m.detach().cpu()[0], dtype=np.float32)
        dyaw = float(vsn_output.dyaw_deg.detach().cpu()[0])
        physical = np.asarray([-self.gain_xy * dxy[0], -self.gain_xy * dxy[1], -self.gain_yaw * dyaw], dtype=np.float32)
        valid = getattr(vsn_output, "valid", None)
        if valid is not None and not bool(valid.detach().cpu()[0]):
            physical[:] = 0.0
            self.last_hold_reason = "invalid_observation"
        if self.confidence_mode == "scale":
            pc = float(vsn_output.position_confidence.detach().cpu()[0])
            oc = float(vsn_output.orientation_confidence.detach().cpu()[0])
            physical *= min(1.0, max(0.0, min(pc, oc) / self.confidence_threshold))
        elif self.confidence_mode == "hold":
            pc = float(vsn_output.position_confidence.detach().cpu()[0])
            oc = float(vsn_output.orientation_confidence.detach().cpu()[0])
            if min(pc, oc) < self.confidence_threshold:
                physical[:] = 0.0
                self.last_hold_reason = "low_confidence"

        if float(np.linalg.norm(physical[:2])) < self.deadband_xy_m:
            physical[:2] = 0.0
        if abs(float(physical[2])) < self.deadband_yaw_deg:
            physical[2] = 0.0

        if self._last_physical is not None:
            active = (np.abs(physical) > 1e-12) & (np.abs(self._last_physical) > 1e-12)
            if bool(np.any(active & (np.sign(physical) != np.sign(self._last_physical)))):
                self.sign_reversals += 1
        if self.max_sign_reversals > 0 and self.sign_reversals > self.max_sign_reversals:
            physical[:] = 0.0
            self.last_hold_reason = "oscillation"

        next_xy = self.cumulative_xy_m + float(np.linalg.norm(physical[:2]))
        next_yaw = self.cumulative_yaw_deg + abs(float(physical[2]))
        if next_xy > self.max_cumulative_xy_m or next_yaw > self.max_cumulative_yaw_deg:
            physical[:] = 0.0
            self.last_hold_reason = "cumulative_motion_limit"
        else:
            self.cumulative_xy_m = next_xy
            self.cumulative_yaw_deg = next_yaw
        self._last_physical = physical.copy()
        return ControllerAction(physical_to_normalized_action(physical, self.max_xy_mm, self.max_yaw_deg), physical)


class OracleController:
    def __init__(self, max_xy_mm=2.0, max_yaw_deg=2.0):
        self.max_xy_mm = max_xy_mm
        self.max_yaw_deg = max_yaw_deg

    def reset(self):
        pass

    def act_from_pose_error(self, pose_error) -> ControllerAction:
        physical = -np.asarray(pose_error, dtype=np.float32)
        return ControllerAction(physical_to_normalized_action(physical, self.max_xy_mm, self.max_yaw_deg), physical)
