"""Measured Panda state contracts and pose helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def wrap_deg(angle: float) -> float:
    return ((float(angle) + 180.0) % 360.0) - 180.0


def pose7(pos, quat) -> np.ndarray:
    return np.asarray([*pos, *quat], dtype=np.float64)


def yaw_from_quat_deg(quat, pybullet_module) -> float:
    euler = pybullet_module.getEulerFromQuaternion(tuple(float(x) for x in quat))
    return math.degrees(float(euler[2]))


@dataclass
class MeasuredPandaState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pos_world: np.ndarray
    ee_quat_world: np.ndarray
    peg_pos_world: np.ndarray
    peg_quat_world: np.ndarray
    peg_tip_pos_world: np.ndarray
    peg_tip_quat_world: np.ndarray
    hole_pos_world: np.ndarray
    hole_quat_world: np.ndarray
    pose_error_task: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data
