"""Panda command/result data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class CartesianDeltaCommand:
    dx_m: float
    dy_m: float
    dyaw_deg: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.dx_m, self.dy_m, self.dyaw_deg], dtype=np.float32)


@dataclass
class ExecutionResult:
    commanded_ee_pose: np.ndarray
    measured_ee_pose: np.ndarray
    commanded_peg_pose: np.ndarray
    measured_peg_pose: np.ndarray
    joint_target: np.ndarray
    joint_actual: np.ndarray
    pos_error_m: float
    yaw_error_deg: float
    max_joint_error: float
    contacts: list[dict[str, Any]]
    ik_success: bool = True
    execution_mode: str = "kinematic"
    joint_limit_violation: bool = False
    joint_limit_margins: np.ndarray | None = None
    ik_residual_m: float | None = None
    ik_branch: str | None = None

    @property
    def pos_error_mm(self) -> float:
        return float(self.pos_error_m) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in [
            "commanded_ee_pose",
            "measured_ee_pose",
            "commanded_peg_pose",
            "measured_peg_pose",
            "joint_target",
            "joint_actual",
        ]:
            data[key] = np.asarray(data[key]).tolist()
        if data["joint_limit_margins"] is not None:
            data["joint_limit_margins"] = np.asarray(data["joint_limit_margins"]).tolist()
        data["pos_error_mm"] = self.pos_error_mm
        return data
