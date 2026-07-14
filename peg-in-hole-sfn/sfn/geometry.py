"""Coordinate, action, and label-codec utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .constants import ORIENTATION_ANGLES_DEG, POSITION_GRID_SIZE, POSITION_RESOLUTION_MM


def encode_position(
    dx_m: float, dy_m: float, grid_size: int = POSITION_GRID_SIZE, resolution_mm: float = POSITION_RESOLUTION_MM
) -> tuple[int, int]:
    center = (grid_size - 1) // 2
    col = int(round(-float(dx_m) * 1000.0 / resolution_mm)) + center
    row = int(round(float(dy_m) * 1000.0 / resolution_mm)) + center
    if not (0 <= row < grid_size and 0 <= col < grid_size):
        raise ValueError(f"Position outside grid: row={row}, col={col}")
    return row, col


def decode_position(
    row: int, col: int, grid_size: int = POSITION_GRID_SIZE, resolution_mm: float = POSITION_RESOLUTION_MM
) -> tuple[float, float]:
    center = (grid_size - 1) // 2
    return (-(int(col) - center) * resolution_mm / 1000.0, (int(row) - center) * resolution_mm / 1000.0)


def encode_position_heatmap(dx_m: float, dy_m: float, grid_size: int = POSITION_GRID_SIZE) -> np.ndarray:
    row, col = encode_position(dx_m, dy_m, grid_size=grid_size)
    hm = np.zeros((grid_size, grid_size), dtype=np.uint8)
    hm[row, col] = 1
    return hm


def encode_orientation(dyaw_deg: float, angles: Sequence[float] = ORIENTATION_ANGLES_DEG) -> int:
    arr = np.asarray(list(angles), dtype=np.float32)
    return int(np.argmin(np.abs(arr - float(dyaw_deg))))


def decode_orientation(index: int, angles: Sequence[float] = ORIENTATION_ANGLES_DEG) -> float:
    return float(list(angles)[int(index)])


def xy_error_mm(pose_error: Sequence[float]) -> float:
    return float(math.hypot(float(pose_error[0]) * 1000.0, float(pose_error[1]) * 1000.0))


def yaw_error_deg(pose_error: Sequence[float]) -> float:
    return abs(float(pose_error[2]))


def is_success(pose_error: Sequence[float], xy_axis_tol_mm: float = 1.0, yaw_tol_deg: float = 2.0) -> bool:
    return (
        abs(float(pose_error[0]) * 1000.0) <= xy_axis_tol_mm
        and abs(float(pose_error[1]) * 1000.0) <= xy_axis_tol_mm
        and abs(float(pose_error[2])) <= yaw_tol_deg
    )


def normalized_to_physical_action(
    action: Sequence[float], max_xy_mm: float = 2.0, max_yaw_deg: float = 2.0
) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
    return np.asarray([a[0] * max_xy_mm / 1000.0, a[1] * max_xy_mm / 1000.0, a[2] * max_yaw_deg], dtype=np.float32)


def physical_to_normalized_action(
    action: Sequence[float], max_xy_mm: float = 2.0, max_yaw_deg: float = 2.0
) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(3)
    return np.clip(
        np.asarray([a[0] * 1000.0 / max_xy_mm, a[1] * 1000.0 / max_xy_mm, a[2] / max_yaw_deg], dtype=np.float32),
        -1.0,
        1.0,
    )


def dense_error_value(
    pose_error: Sequence[float],
    xy_range_mm: float = 15.0,
    yaw_range_deg: float = 15.0,
    w_xy: float = 0.7,
    w_yaw: float = 0.3,
) -> float:
    return float(
        w_xy * xy_error_mm(pose_error) / float(xy_range_mm) + w_yaw * yaw_error_deg(pose_error) / float(yaw_range_deg)
    )
