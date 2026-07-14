"""Panda inverse-kinematics helpers."""

from __future__ import annotations

import numpy as np


def joint_limit_arrays(pybullet_module, robot_id: int, joint_indices: tuple[int, ...], physics_client_id: int):
    lowers, uppers, ranges = [], [], []
    for joint_index in joint_indices:
        info = pybullet_module.getJointInfo(robot_id, int(joint_index), physicsClientId=physics_client_id)
        lo, hi = float(info[8]), float(info[9])
        if hi <= lo:
            lo, hi = -3.14159, 3.14159
        lowers.append(lo)
        uppers.append(hi)
        ranges.append(hi - lo)
    return lowers, uppers, ranges


def clamp_joints(values, lowers, uppers) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    return np.minimum(np.maximum(v, np.asarray(lowers, dtype=np.float64)), np.asarray(uppers, dtype=np.float64))
