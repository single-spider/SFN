"""Standalone-object insertion environment built on the alignment env.

No robot arm, IK, ROS, or hardware middleware is used.  The peg remains a
kinematic standalone object; once alignment success is reached the environment
performs a deterministic Z descent and validates residual pose tolerances.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..config import InsertionConfig
from ..geometry import is_success, xy_error_mm, yaw_error_deg
from .alignment_env import PegInHoleAlignmentEnv
from .mesh_insertion import simulate_mesh_insertion


class PegInHoleInsertionEnv(PegInHoleAlignmentEnv):
    """ALIGN -> DESCEND -> SUCCESS/FAILURE insertion task.

    ``collision_mode="proxy"`` checks residual insertion tolerances only.
    ``collision_mode="geometric"`` is deterministic and currently uses the same
    tolerance gate plus a synthetic penetration proxy.  It is intentionally
    isolated here so a future PyBullet contact query can replace the proxy
    without changing the public env contract.
    """

    def __init__(self, *args, insertion_config: InsertionConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.insertion_config = insertion_config or InsertionConfig()
        self.phase = "ALIGN"
        self.depth_mm = 0.0
        self._last_insertion_info: dict | None = None

    def reset(self, *, seed=None, options=None):
        self.phase = "ALIGN"
        self.depth_mm = 0.0
        self._last_insertion_info = None
        obs, info = super().reset(seed=seed, options=options)
        info["phase"] = self.phase
        info["insertion_config"] = asdict(self.insertion_config)
        return obs, info

    def step(self, action):
        if self.phase in {"SUCCESS", "FAILURE"}:
            raise RuntimeError("Episode has already completed; call reset().")

        obs, reward, terminated, truncated, info = super().step(action)
        info["phase"] = self.phase

        # Alignment success is necessary but not sufficient for insertion.
        if info["success"]:
            self.phase = "DESCEND"
            inserted, extra = self.attempt_insertion()
            info.update(extra)
            terminated = True
            truncated = False
            reward += 1.0 if inserted else -1.0
            info["success"] = bool(inserted)
            info["termination_reason"] = "insertion_success" if inserted else "insertion_failure"
        elif terminated:
            # Out-of-bounds alignment failure.
            self.phase = "FAILURE"
            info["phase"] = self.phase
            info["termination_reason"] = "alignment_failure"
        elif truncated:
            info["termination_reason"] = "max_steps"
        return obs, reward, terminated, truncated, info

    def attempt_insertion(self) -> tuple[bool, dict]:
        """Run deterministic descent from the current pose and return outcome.

        Public on purpose: tests and evaluators can validate exact-alignment and
        intentional-misalignment insertion behavior without faking robot motion.
        """
        assert self.state is not None

        if self.insertion_config.collision_mode == "geometric" and self.state.shape != "synthetic-square":
            result = simulate_mesh_insertion(
                self.state.shape,
                self.state.pose_error,
                self.insertion_config,
                self.registry,
            )
            self.depth_mm = result.insertion_depth_mm
            self.phase = "SUCCESS" if result.success else "FAILURE"
            info = result.to_dict()
            info.update(
                {
                    "phase": self.phase,
                    "residual_xy_error_mm": xy_error_mm(self.state.pose_error),
                    "residual_yaw_error_deg": yaw_error_deg(self.state.pose_error),
                    "insertion_residual_ok": bool(
                        is_success(
                            self.state.pose_error,
                            self.insertion_config.insertion_xy_axis_mm,
                            self.insertion_config.insertion_yaw_deg,
                        )
                    ),
                    "reached_insertion_depth": bool(result.reached_depth),
                    "collision_mode": "geometric",
                }
            )
            self._last_insertion_info = info
            return result.success, info

        attempts = 0
        self.depth_mm = 0.0
        while (
            attempts < self.insertion_config.max_descent_attempts
            and self.depth_mm + 1e-9 < self.insertion_config.target_depth_mm
        ):
            attempts += 1
            self.depth_mm = min(
                self.insertion_config.target_depth_mm, self.depth_mm + self.insertion_config.descent_increment_mm
            )

        residual_ok = is_success(
            self.state.pose_error,
            self.insertion_config.insertion_xy_axis_mm,
            self.insertion_config.insertion_yaw_deg,
        )
        reached_depth = self.depth_mm + 1e-9 >= self.insertion_config.target_depth_mm
        collision_failure, penetration_mm = self._collision_proxy()
        success = bool(reached_depth and residual_ok and not collision_failure)
        self.phase = "SUCCESS" if success else "FAILURE"

        info = {
            "phase": self.phase,
            "insertion_depth_mm": float(self.depth_mm),
            "target_depth_mm": float(self.insertion_config.target_depth_mm),
            "descent_attempts": attempts,
            "residual_xy_error_mm": xy_error_mm(self.state.pose_error),
            "residual_yaw_error_deg": yaw_error_deg(self.state.pose_error),
            "insertion_residual_ok": bool(residual_ok),
            "reached_insertion_depth": bool(reached_depth),
            "collision_mode": self.insertion_config.collision_mode,
            "collision_failure": bool(collision_failure),
            "penetration_proxy_mm": float(penetration_mm),
        }
        self._last_insertion_info = info
        return success, info

    def _collision_proxy(self) -> tuple[bool, float]:
        """Deterministic contact proxy for dependency-light Milestone B.

        If residual pose exceeds insertion tolerances, report the excess as
        synthetic penetration.  In ``proxy`` mode this is still recorded but does
        not independently fail beyond the residual tolerance gate; in
        ``geometric`` mode it is also exposed as ``collision_failure``.
        """
        assert self.state is not None
        dx_excess = max(0.0, abs(float(self.state.pose_error[0])) * 1000.0 - self.insertion_config.insertion_xy_axis_mm)
        dy_excess = max(0.0, abs(float(self.state.pose_error[1])) * 1000.0 - self.insertion_config.insertion_xy_axis_mm)
        yaw_excess = max(0.0, abs(float(self.state.pose_error[2])) - self.insertion_config.insertion_yaw_deg)
        penetration_mm = float(np.hypot(dx_excess, dy_excess) + yaw_excess * 0.1)
        if self.insertion_config.collision_mode == "proxy":
            return False, penetration_mm
        return penetration_mm > self.insertion_config.max_collision_penetration_mm, penetration_mm
