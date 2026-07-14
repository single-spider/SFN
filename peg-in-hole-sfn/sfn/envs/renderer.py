"""Deterministic toy and asset-faithful Cartesian renderers.

``SyntheticDirectRenderer`` is intentionally retained as a cheap legacy/toy
backend.  It does *not* represent the supplied shape meshes and must not be used
for final unseen-shape claims.

``MeshOrthographicRenderer`` projects the real ``peg.obj`` and ``mask.obj``
triangles into a calibrated top-down crop.  The semantic mask follows the SFN
contract: background=0, peg=1, and visible hole/seam region=2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CameraConfig
from ..constants import MASK_BACKGROUND, MASK_PEG, MASK_SEAM
from .asset_registry import AssetRegistry


@dataclass
class RenderOutput:
    rgb: np.ndarray
    mask: np.ndarray
    metadata: dict[str, Any] | None = None


class SyntheticDirectRenderer:
    def __init__(self, camera: CameraConfig | None = None):
        self.camera = camera or CameraConfig()
        self.crop_h = int(self.camera.crop_height)
        self.crop_w = int(self.camera.crop_width)

    def render(self, pose_error: np.ndarray, shape: str, rng: np.random.Generator | None = None) -> RenderOutput:
        h, w = self.crop_h, self.crop_w
        mask = np.zeros((h, w), dtype=np.uint8)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:] = [36, 36, 36]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w // 2, h // 2
        hole = (np.abs(xx - cx) <= 42) & (np.abs(yy - cy) <= 34)
        mask[hole] = MASK_SEAM
        rgb[hole] = [170, 170, 170]
        dx, dy, dyaw = map(float, pose_error)
        pcx = int(round(cx - dx * 1000.0))
        pcy = int(round(cy + dy * 1000.0))
        th = math.radians(dyaw)
        c, s = math.cos(th), math.sin(th)
        x0 = xx - pcx
        y0 = yy - pcy
        xr = c * x0 + s * y0
        yr = -s * x0 + c * y0
        tweak = (sum(ord(ch) for ch in shape) % 9) - 4
        half_w = 30 + tweak
        half_h = 28 - tweak // 2
        # Avoid perfectly square pegs.  A square mask has no visual yaw signal,
        # which made the orientation validation target mathematically
        # impossible for some shape names (notably square-diamond).
        if abs(half_w - half_h) < 3:
            half_h = max(20, half_h - 3)
        peg = (np.abs(xr) <= half_w) & (np.abs(yr) <= half_h)
        mask[peg] = MASK_PEG
        rgb[peg] = [35, 210, 55]
        return RenderOutput(
            np.transpose(rgb, (2, 0, 1)).astype(np.uint8),
            mask,
            {"renderer_backend": "toy_direct", "asset_faithful": False},
        )

    def close(self):
        pass


class MeshOrthographicRenderer:
    """Rasterize actual peg and opening meshes into an orthographic crop.

    The renderer deliberately performs semantic render passes rather than
    inferring labels from RGB colours.  ``mask.obj`` describes the nominal hole
    opening.  The visible seam is the opening silhouette not occluded by the
    transformed peg silhouette, matching the legacy SFN mask construction.
    """

    def __init__(self, camera: CameraConfig | None = None, asset_registry: AssetRegistry | None = None):
        self.camera = camera or CameraConfig(renderer_backend="mesh_orthographic")
        self.registry = asset_registry or AssetRegistry()
        self.crop_h = int(self.camera.crop_height)
        self.crop_w = int(self.camera.crop_width)
        self.pixels_per_mm = float(self.camera.orthographic_pixels_per_mm)
        self._mesh_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    @property
    def backend_name(self) -> str:
        return "mesh_orthographic"

    def _load_mesh_arrays(self, shape: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._mesh_cache.get(shape)
        if cached is not None:
            return cached
        assets = self.registry.get(shape)
        try:
            import trimesh
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("trimesh is required for mesh_orthographic rendering") from exc

        def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
            mesh = trimesh.load_mesh(str(path), force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"Expected triangular mesh at {path}")
            if not np.isfinite(vertices).all() or len(faces) == 0:
                raise ValueError(f"Invalid or empty mesh at {path}")
            return vertices, faces

        peg_v, peg_f = load(assets.peg_obj)
        hole_v, hole_f = load(assets.mask_obj)
        cached = (peg_v, peg_f, hole_v, hole_f)
        self._mesh_cache[shape] = cached
        return cached

    def _task_xy_to_pixels(self, xy_m: np.ndarray) -> np.ndarray:
        """Map task XY to image pixels using the established SFN signs."""
        xy = np.asarray(xy_m, dtype=np.float64)
        out = np.empty_like(xy, dtype=np.float64)
        out[..., 0] = self.crop_w / 2.0 - xy[..., 0] * 1000.0 * self.pixels_per_mm
        out[..., 1] = self.crop_h / 2.0 + xy[..., 1] * 1000.0 * self.pixels_per_mm
        return out

    def _rasterize(self, vertices_xy_m: np.ndarray, faces: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
            raise RuntimeError("opencv-python is required for mesh_orthographic rendering") from exc
        pixels = self._task_xy_to_pixels(vertices_xy_m)
        canvas = np.zeros((self.crop_h, self.crop_w), dtype=np.uint8)
        # Projecting every triangle and taking their union produces the exact
        # top-down silhouette for these extruded mesh assets.
        for face in faces:
            tri = np.rint(pixels[face]).astype(np.int32)
            edge_a = tri[1] - tri[0]
            edge_b = tri[2] - tri[0]
            signed_area2 = float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
            if abs(signed_area2) < 0.5:
                continue
            cv2.fillConvexPoly(canvas, tri, 1, lineType=cv2.LINE_8)
        return canvas.astype(bool)

    def render(self, pose_error: np.ndarray, shape: str, rng: np.random.Generator | None = None) -> RenderOutput:
        del rng  # deterministic backend; randomization is a separate recorded transform
        peg_v, peg_f, hole_v, hole_f = self._load_mesh_arrays(shape)
        dx_m, dy_m, dyaw_deg = map(float, np.asarray(pose_error).reshape(3))
        theta = math.radians(dyaw_deg)
        rot = np.asarray([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=np.float64)
        peg_xy = peg_v[:, :2] @ rot.T + np.asarray([dx_m, dy_m], dtype=np.float64)
        hole_xy = hole_v[:, :2]
        peg = self._rasterize(peg_xy, peg_f)
        hole = self._rasterize(hole_xy, hole_f)
        seam = hole & ~peg

        mask = np.full((self.crop_h, self.crop_w), MASK_BACKGROUND, dtype=np.uint8)
        mask[seam] = MASK_SEAM
        mask[peg] = MASK_PEG

        rgb = np.empty((self.crop_h, self.crop_w, 3), dtype=np.uint8)
        rgb[:] = [36, 36, 36]
        # A neutral fixture patch gives segmentation a nontrivial local context
        # while labels remain generated from geometry, never colour thresholds.
        base_half_px = int(round(15.0 * self.pixels_per_mm))
        cx, cy = self.crop_w // 2, self.crop_h // 2
        y0, y1 = max(0, cy - base_half_px), min(self.crop_h, cy + base_half_px + 1)
        x0, x1 = max(0, cx - base_half_px), min(self.crop_w, cx + base_half_px + 1)
        rgb[y0:y1, x0:x1] = [105, 105, 108]
        rgb[hole] = [170, 170, 170]
        rgb[peg] = [35, 210, 55]

        metadata = {
            "renderer_backend": self.backend_name,
            "asset_faithful": True,
            "shape": shape,
            "pixels_per_mm": self.pixels_per_mm,
            "peg_pixels": int(peg.sum()),
            "hole_pixels": int(hole.sum()),
            "seam_pixels": int(seam.sum()),
        }
        return RenderOutput(np.transpose(rgb, (2, 0, 1)), mask, metadata)

    def close(self) -> None:
        self._mesh_cache.clear()


def make_renderer(
    camera: CameraConfig | None = None,
    asset_registry: AssetRegistry | None = None,
):
    """Create the explicitly configured renderer backend."""
    camera = camera or CameraConfig()
    if camera.renderer_backend == "toy_direct":
        return SyntheticDirectRenderer(camera)
    if camera.renderer_backend == "mesh_orthographic":
        return MeshOrthographicRenderer(camera, asset_registry=asset_registry)
    raise ValueError(f"Unsupported renderer backend: {camera.renderer_backend!r}")
