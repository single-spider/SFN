"""Metric task-plane rectification for calibrated Panda cameras."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PandaConfig, TaskToWorldTransform


@dataclass(frozen=True)
class CanonicalCamera:
    width: int = 500
    height: int = 500
    extent_x_mm: float = 50.0
    extent_y_mm: float = 50.0
    plane_z_m: float = 0.00075


class PandaCameraCanonicalizer:
    """Warp an oblique perspective image to a metric top-down task plane."""

    def __init__(
        self,
        panda: PandaConfig,
        source_width: int,
        source_height: int,
        fov_y_deg: float,
        canonical: CanonicalCamera | None = None,
        transform: TaskToWorldTransform | None = None,
    ):
        import cv2
        import pybullet as p

        self.config = canonical or CanonicalCamera()
        task = transform or TaskToWorldTransform()
        origin = np.asarray(task.origin_world, dtype=np.float64)
        eye = origin + np.asarray(panda.camera_eye_offset_m, dtype=np.float64)
        target = origin + np.asarray(panda.camera_target_offset_m, dtype=np.float64)
        view = np.asarray(
            p.computeViewMatrix(eye.tolist(), target.tolist(), list(panda.camera_up_vector)), dtype=np.float64
        ).reshape(4, 4, order="F")
        proj = np.asarray(
            p.computeProjectionMatrixFOV(float(fov_y_deg), float(source_width) / source_height, 0.001, 1.0),
            dtype=np.float64,
        ).reshape(4, 4, order="F")
        ex, ey = self.config.extent_x_mm / 2000.0, self.config.extent_y_mm / 2000.0
        offsets = np.asarray([[-ex, -ey], [ex, -ey], [ex, ey], [-ex, ey]], dtype=np.float64)
        source = []
        for dx, dy in offsets:
            world = task.task_pose_to_world_pos(float(dx), float(dy), self.config.plane_z_m)
            clip = proj @ view @ np.r_[world, 1.0]
            ndc = clip[:3] / clip[3]
            source.append([(ndc[0] + 1.0) * 0.5 * source_width, (1.0 - ndc[1]) * 0.5 * source_height])
        destination = np.asarray(
            [
                [0, self.config.height - 1],
                [self.config.width - 1, self.config.height - 1],
                [self.config.width - 1, 0],
                [0, 0],
            ],
            dtype=np.float32,
        )
        self.matrix = cv2.getPerspectiveTransform(np.asarray(source, dtype=np.float32), destination)

    def warp_mask(self, mask: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.warpPerspective(
            np.asarray(mask, dtype=np.uint8),
            self.matrix,
            (self.config.width, self.config.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def warp_rgb(self, rgb_chw: np.ndarray) -> np.ndarray:
        import cv2

        hwc = np.transpose(np.asarray(rgb_chw, dtype=np.uint8), (1, 2, 0))
        warped = cv2.warpPerspective(
            hwc,
            self.matrix,
            (self.config.width, self.config.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return np.transpose(warped, (2, 0, 1)).astype(np.uint8)
