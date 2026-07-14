"""Asset-faithful geometric Z-insertion using real mesh silhouettes/bounds."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from ..config import CameraConfig, InsertionConfig
from .asset_registry import AssetRegistry
from .renderer import MeshOrthographicRenderer


@dataclass
class MeshInsertionResult:
    success: bool
    reached_depth: bool
    collision_failure: bool
    insertion_depth_mm: float
    target_depth_mm: float
    descent_attempts: int
    outside_pixels: int
    outside_area_mm2: float
    penetration_proxy_mm: float
    peg_origin_start_z_mm: float
    peg_origin_final_z_mm: float
    peg_tip_start_z_mm: float
    peg_tip_final_z_mm: float
    base_top_z_mm: float
    renderer_backend: str = "mesh_geometric_insertion"

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_mesh_insertion(
    shape: str,
    pose_error: np.ndarray,
    config: InsertionConfig,
    asset_registry: AssetRegistry | None = None,
) -> MeshInsertionResult:
    """Descend a mesh-derived peg until target depth or first rim collision.

    This is a deterministic geometric backend rather than a force-dynamics
    model. Unlike the old proxy, it moves the peg reference in Z using the real
    mesh bounds and determines collision from the actual peg/opening
    silhouettes. Panda dynamic/contact insertion remains a separate later gate.
    """
    registry = asset_registry or AssetRegistry()
    ppm = float(config.geometry_pixels_per_mm)
    extent_mm = 50.0
    crop = int(math.ceil(extent_mm * ppm))
    if crop % 2:
        crop += 1
    camera = CameraConfig(
        crop_width=crop,
        crop_height=crop,
        renderer_backend="mesh_orthographic",
        orthographic_pixels_per_mm=ppm,
    )
    renderer = MeshOrthographicRenderer(camera, registry)
    peg_v, peg_f, hole_v, hole_f = renderer._load_mesh_arrays(shape)

    pose = np.asarray(pose_error, dtype=np.float64).reshape(3)
    theta = math.radians(float(pose[2]))
    rot = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    peg_xy = peg_v[:, :2] @ rot.T + pose[:2]
    peg_mask = renderer._rasterize(peg_xy, peg_f)
    hole_mask = renderer._rasterize(hole_v[:, :2], hole_f)
    outside = peg_mask & ~hole_mask
    outside_pixels = int(outside.sum())
    outside_area_mm2 = float(outside_pixels / (ppm * ppm))

    penetration_proxy_mm = 0.0
    if outside_pixels:
        try:
            import cv2

            outside_region = (~hole_mask).astype(np.uint8)
            distance_px = cv2.distanceTransform(outside_region, cv2.DIST_L2, 5)
            penetration_proxy_mm = float(distance_px[outside].max() / ppm)
        except (ModuleNotFoundError, ValueError):  # pragma: no cover - diagnostic fallback
            penetration_proxy_mm = float(math.sqrt(outside_area_mm2))

    base_mesh = registry.get(shape).base_obj
    try:
        import trimesh
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("trimesh is required for geometric insertion") from exc
    base = trimesh.load_mesh(str(base_mesh), force="mesh", process=False)
    base_top_z_m = float(np.max(np.asarray(base.vertices)[:, 2]))
    peg_tip_local_z_m = float(np.min(peg_v[:, 2]))
    start_tip_z_m = base_top_z_m + float(config.approach_clearance_mm) / 1000.0
    start_origin_z_m = start_tip_z_m - peg_tip_local_z_m

    depth_mm = 0.0
    attempts = 0
    collision = False
    while attempts < int(config.max_descent_attempts) and depth_mm + 1e-9 < float(config.target_depth_mm):
        attempts += 1
        proposed_depth = min(
            float(config.target_depth_mm),
            depth_mm + float(config.descent_increment_mm),
        )
        proposed_tip_z_m = start_tip_z_m - proposed_depth / 1000.0
        overlaps_fixture = proposed_tip_z_m < base_top_z_m - 1e-12
        if overlaps_fixture and outside_pixels > 0:
            collision = True
            break
        depth_mm = proposed_depth

    reached = depth_mm + 1e-9 >= float(config.target_depth_mm)
    final_origin_z_m = start_origin_z_m - depth_mm / 1000.0
    final_tip_z_m = start_tip_z_m - depth_mm / 1000.0
    return MeshInsertionResult(
        success=bool(reached and not collision),
        reached_depth=bool(reached),
        collision_failure=bool(collision),
        insertion_depth_mm=float(depth_mm),
        target_depth_mm=float(config.target_depth_mm),
        descent_attempts=int(attempts),
        outside_pixels=outside_pixels,
        outside_area_mm2=outside_area_mm2,
        penetration_proxy_mm=penetration_proxy_mm,
        peg_origin_start_z_mm=start_origin_z_m * 1000.0,
        peg_origin_final_z_mm=final_origin_z_m * 1000.0,
        peg_tip_start_z_mm=start_tip_z_m * 1000.0,
        peg_tip_final_z_mm=final_tip_z_m * 1000.0,
        base_top_z_mm=base_top_z_m * 1000.0,
    )
