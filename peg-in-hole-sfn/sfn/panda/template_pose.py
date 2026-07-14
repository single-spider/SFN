"""Asset-template pose estimation for a calibrated top-down Panda camera."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import PandaConfig, TaskToWorldTransform, default_asset_root


class PandaTopdownTemplatePoseEstimator:
    """Estimate task XY/yaw by matching the known peg mesh silhouette.

    The fixture/peg asset is task setup information, not ground-truth pose.
    Camera extrinsics map pixel translation to task-frame metric translation;
    exhaustive small-angle silhouette correlation supplies yaw.
    """

    def __init__(
        self,
        shape: str,
        panda_config: PandaConfig,
        width: int = 500,
        height: int = 400,
        fov_y_deg: float = 35.0,
        angles=None,
        asset_root: str | Path | None = None,
    ):
        import cv2
        import pybullet as p
        import trimesh

        self.shape = shape
        self.width = width
        self.height = height
        self.angles = np.asarray(angles if angles is not None else np.arange(-10, 10.0001, 0.25), dtype=np.float64)
        task = TaskToWorldTransform()
        origin = np.asarray(task.origin_world, dtype=np.float64)
        eye = origin + np.asarray(panda_config.camera_eye_offset_m)
        target = origin + np.asarray(panda_config.camera_target_offset_m)
        self.view = np.asarray(
            p.computeViewMatrix(eye.tolist(), target.tolist(), list(panda_config.camera_up_vector))
        ).reshape(4, 4, order="F")
        self.proj = np.asarray(p.computeProjectionMatrixFOV(fov_y_deg, width / height, 0.001, 1.0)).reshape(
            4, 4, order="F"
        )
        root = Path(asset_root) if asset_root else default_asset_root()
        mesh = trimesh.load_mesh(str(root / shape / "peg" / "peg.obj"), force="mesh", process=False)
        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(mesh.faces, dtype=np.int32)
        self.peg_origin_z = 0.0605
        # Pixel Jacobian d(pixel_xy)/d(task_xy_m) at the peg top plane.
        c = self._project_world(task.task_pose_to_world_pos(0, 0, self.peg_origin_z))
        px = self._project_world(task.task_pose_to_world_pos(0.001, 0, self.peg_origin_z))
        py = self._project_world(task.task_pose_to_world_pos(0, 0.001, self.peg_origin_z))
        self.center_pixel = c
        self.jacobian = np.column_stack(((px - c) / 0.001, (py - c) / 0.001))
        self.inv_jacobian = np.linalg.inv(self.jacobian)
        templates = []
        centroids = []
        for angle in self.angles:
            theta = math.radians(float(angle))
            rot = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            xy = self.vertices[:, :2] @ rot.T
            screen = np.asarray(
                [
                    self._project_world(origin + np.asarray([x, y, self.peg_origin_z + z]))
                    for (x, y), z in zip(xy, self.vertices[:, 2], strict=False)
                ]
            )
            canvas = np.zeros((height, width), np.uint8)
            for face in self.faces:
                cv2.fillConvexPoly(canvas, np.rint(screen[face]).astype(np.int32), 1)
            centroid = self._centroid(canvas)
            centroids.append(centroid)
            templates.append(self._center_crop(canvas, centroid, 128))
        self.templates = np.asarray(templates, dtype=bool)
        self.template_centroids = np.asarray(centroids)

    def _project_world(self, world):
        clip = self.proj @ self.view @ np.r_[world, 1.0]
        ndc = clip[:3] / clip[3]
        return np.asarray([(ndc[0] + 1) * 0.5 * self.width, (1 - ndc[1]) * 0.5 * self.height])

    @staticmethod
    def _centroid(mask):
        yy, xx = np.nonzero(mask)
        return np.asarray([xx.mean(), yy.mean()]) if len(xx) else np.asarray([np.nan, np.nan])

    @staticmethod
    def _center_crop(mask, centroid, size):
        import cv2

        target = (size - 1) / 2.0
        matrix = np.asarray([[1.0, 0.0, target - centroid[0]], [0.0, 1.0, target - centroid[1]]])
        return cv2.warpAffine(mask.astype(np.uint8), matrix, (size, size), flags=cv2.INTER_NEAREST) > 0

    def estimate(self, semantic_mask):
        peg = np.asarray(semantic_mask) == 1
        centroid = self._centroid(peg)
        if not np.isfinite(centroid).all() or int(peg.sum()) < 20:
            return np.zeros(2), 0.0, 0.0, False
        observed = self._center_crop(peg, centroid, 128)
        inter = np.logical_and(self.templates, observed).sum((1, 2))
        union = np.logical_or(self.templates, observed).sum((1, 2))
        scores = inter / np.maximum(union, 1)
        index = int(np.argmax(scores))
        yaw = float(self.angles[index])
        pixel_delta = centroid - self.template_centroids[index]
        dxy = self.inv_jacobian @ pixel_delta
        sorted_scores = np.sort(scores)
        confidence = float(
            scores[index]
            * (1.0 if len(scores) < 2 else max(0.05, 1 - (sorted_scores[-2] / max(sorted_scores[-1], 1e-9)) * 0.5))
        )
        return dxy, yaw, confidence, True
