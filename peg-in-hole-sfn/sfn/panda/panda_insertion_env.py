"""Measured, incremental Panda peg insertion.

Alignment remains an ordinary XY/yaw environment step.  Once alignment is
accepted, the controller advances the *measured* end-effector pose in small Z
increments and evaluates insertion from the measured peg-tip pose.  This keeps
the same code path usable in explicit kinematic (geometry/IK validation) and
dynamic (motor/contact validation) scene modes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from ..config import EnvironmentConfig, InsertionConfig
from ..geometry import is_success
from .panda_alignment_env import PandaPegInHoleAlignmentEnv


class PandaPegInHoleInsertionEnv(PandaPegInHoleAlignmentEnv):
    """Alignment followed by a bounded incremental vertical descent."""

    _STALL_ATTEMPTS = 3

    def __init__(
        self,
        *args,
        insertion_config: InsertionConfig | None = None,
        insertion_observer: Callable[[dict, dict], None] | None = None,
        **kwargs,
    ):
        self.insertion_config = insertion_config or InsertionConfig()
        if insertion_observer is not None and not callable(insertion_observer):
            raise TypeError("insertion_observer must be callable")
        self.insertion_observer = insertion_observer
        # The alignment phase must continue until the *insertion* tolerance is
        # met.  Terminating at the looser generic alignment threshold leaves a
        # controller no opportunity to make the final sub-millimetre update.
        env_config = kwargs.get("env_config") or EnvironmentConfig()
        kwargs["env_config"] = replace(
            env_config,
            task="insertion",
            xy_success_axis_mm=float(self.insertion_config.insertion_xy_axis_mm),
            yaw_success_deg=float(self.insertion_config.insertion_yaw_deg),
        )
        # Physical insertion must begin with the mesh-derived peg tip just
        # above the fixture; the legacy z=2 mm pose already penetrates the
        # 60 mm peg through the base and makes measured-depth checks meaningless.
        panda_config = kwargs.get("panda_config")
        if panda_config is not None and not panda_config.mesh_derived_alignment_z:
            kwargs["panda_config"] = replace(panda_config, mesh_derived_alignment_z=True)
        super().__init__(*args, **kwargs)

    def _tip_depth_mm(self) -> float:
        """Depth of the measured tip below the measured base top plane."""
        assert self.scene is not None
        state = self.scene.measure()
        base_top_world_z = float(state.hole_pos_world[2] + self.scene.base_top_z_m)
        return max(0.0, (base_top_world_z - float(state.peg_tip_pos_world[2])) * 1000.0)

    def _descend_once(self, dz_m: float):
        """Command one world-Z increment from the current measured EE pose."""
        assert self.scene is not None
        measured = self.scene.measure()
        target = measured.ee_pos_world.copy()
        target[2] -= float(dz_m)
        joints = self.scene.solve_ik(target, measured.ee_quat_world, measured.joint_positions)
        # Preserve the measured planar command contract; execute_joint_target
        # handles kinematic reset/sync versus dynamic motor simulation.
        return self.scene.execute_joint_target(
            joints,
            measured.pose_error_task,
            # An insertion observer already records one measured sample per
            # completed descent increment.  Emitting the lower-level motor
            # settling samples as well interleaves zero-depth alignment frames
            # with insertion frames, making a monotonic descent look like an
            # up/down hammering motion in exported replays.
            physics_observer=None if self.insertion_observer is not None else self.motion_observer,
            observer_stride=self.motion_observer_stride,
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(
            insertion_success=False,
            insertion_depth_mm=self._tip_depth_mm(),
            insertion_attempts=0,
            insertion_contacts=[],
            max_contact_force=0.0,
            lateral_drift_mm=0.0,
            collision_failure=False,
            jammed=False,
            execution_mode=self.panda_config.execution_mode,
            insertion_trace=[],
        )

        # The parent terminates when alignment succeeds.  It truncates when the
        # alignment horizon is exhausted; insertion must not erase that result.
        if not info["success"]:
            info["termination_reason"] = "panda_alignment_timeout" if truncated else "panda_alignment_incomplete"
            info["failure_state"] = info["termination_reason"]
            return obs, reward, terminated, truncated, info
        if not is_success(
            info["pose_error"],
            self.insertion_config.insertion_xy_axis_mm,
            self.insertion_config.insertion_yaw_deg,
        ):
            info["collision_failure"] = True
            info["termination_reason"] = "panda_insertion_alignment_out_of_tolerance"
            info["failure_state"] = info["termination_reason"]
            return obs, reward - float(self.config.failure_penalty), True, False, info

        assert self.scene is not None
        start_xy = self.scene.measure().pose_error_task[:2].copy()
        previous_pose = self.scene.measure().pose_error_task.copy()
        previous_depth = self._tip_depth_mm()
        stalled = 0
        all_contacts: list[dict] = []
        reason = "panda_insertion_timeout"
        increment_m = float(self.insertion_config.descent_increment_mm) / 1000.0

        for attempt in range(1, int(self.insertion_config.max_descent_attempts) + 1):
            try:
                result = self._descend_once(increment_m)
            except (RuntimeError, ValueError):
                reason = "panda_insertion_ik_failure"
                info["insertion_attempts"] = attempt
                info["insertion_trace"].append({"attempt": attempt, "failure_state": reason})
                break

            state = self.scene.measure()
            depth = self._tip_depth_mm()
            contacts = list(result.contacts or self.scene.contact_summary())
            all_contacts.extend(contacts)
            drift = float(np.linalg.norm(state.pose_error_task[:2] - start_xy) * 1000.0)
            max_force = max((float(c.get("normal_force", 0.0)) for c in all_contacts), default=0.0)
            max_penetration = max(
                (max(0.0, -float(c.get("distance", 0.0))) * 1000.0 for c in all_contacts), default=0.0
            )
            margins = self.scene.joint_limit_margins(state.joint_positions)
            info["insertion_trace"].append(
                {
                    "attempt": attempt,
                    "commanded_dx_m": 0.0,
                    "commanded_dy_m": 0.0,
                    "commanded_dz_m": -increment_m,
                    "commanded_dyaw_deg": 0.0,
                    "measured_dx_m": float(state.pose_error_task[0] - previous_pose[0]),
                    "measured_dy_m": float(state.pose_error_task[1] - previous_pose[1]),
                    "measured_dz_m": float(-(depth - previous_depth) / 1000.0),
                    "measured_dyaw_deg": float(
                        ((state.pose_error_task[2] - previous_pose[2] + 180.0) % 360.0) - 180.0
                    ),
                    "joint_positions": state.joint_positions.tolist(),
                    "joint_target": result.joint_target.tolist(),
                    "joint_limit_margins": margins.tolist(),
                    "min_joint_limit_margin": float(np.min(margins)),
                    "ik_residual_m": result.ik_residual_m,
                    "ik_branch": result.ik_branch,
                    "contact_count": len(contacts),
                    "contacts": contacts,
                    "max_contact_force": max((float(c.get("normal_force", 0.0)) for c in contacts), default=0.0),
                    "max_penetration_mm": max(
                        (max(0.0, -float(c.get("distance", 0.0))) * 1000.0 for c in contacts), default=0.0
                    ),
                    "lateral_drift_mm": drift,
                    "insertion_depth_mm": depth,
                    "failure_state": None,
                }
            )
            info.update(
                insertion_attempts=attempt,
                insertion_depth_mm=depth,
                insertion_contacts=all_contacts,
                contact_count=len(contacts),
                max_contact_force=max_force,
                max_penetration_mm=max_penetration,
                lateral_drift_mm=drift,
                joint_positions=state.joint_positions.astype(np.float32),
                joint_target=result.joint_target.astype(np.float32),
                joint_limit_margins=margins.astype(np.float32),
                min_joint_limit_margin=float(np.min(margins)),
                joint_limit_violation=bool(np.any(margins < -1e-7)),
                ik_residual_m=result.ik_residual_m,
                ik_branch=result.ik_branch,
                contacts=all_contacts,
            )
            if self.insertion_observer is not None:
                # Rendering is opt-in because native-camera capture is relatively
                # expensive.  The observer gets the real post-command observation
                # and a detached telemetry record for this descent attempt.
                self.insertion_observer(self._make_obs(), dict(info["insertion_trace"][-1]))

            if drift > float(self.insertion_config.insertion_xy_axis_mm):
                reason = "panda_insertion_lateral_drift"
                info["collision_failure"] = True
                break
            if depth >= float(self.insertion_config.target_depth_mm):
                info["insertion_success"] = True
                reason = "panda_insertion_success"
                reward += float(self.config.success_bonus)
                break

            progress = depth - previous_depth
            stalled = stalled + 1 if progress < 0.1 * float(self.insertion_config.descent_increment_mm) else 0
            previous_depth = depth
            previous_pose = state.pose_error_task.copy()
            if contacts and stalled >= self._STALL_ATTEMPTS:
                reason = "panda_insertion_jam"
                info["jammed"] = True
                info["collision_failure"] = True
                break

        info["termination_reason"] = reason
        info["failure_state"] = None if info["insertion_success"] else reason
        if info["insertion_trace"]:
            info["insertion_trace"][-1]["failure_state"] = info["failure_state"]
        if not info["insertion_success"]:
            reward -= float(self.config.failure_penalty)
        # Refresh observation after descent; insertion is one bounded macro-step.
        return self._make_obs(), reward, True, False, info
