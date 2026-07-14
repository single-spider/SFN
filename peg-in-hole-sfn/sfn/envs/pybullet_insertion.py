"""Standalone PyBullet collision/contact insertion using supplied meshes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..config import InsertionConfig
from .asset_registry import AssetRegistry


@dataclass
class PyBulletInsertionResult:
    success: bool
    reason: str
    measured_depth_mm: float
    target_depth_mm: float
    attempts: int
    contact_count: int
    max_normal_force_n: float
    max_penetration_mm: float
    lateral_drift_mm: float
    renderer_backend: str = "standalone_pybullet_raster_compound"

    def to_dict(self) -> dict:
        return asdict(self)


def _mask_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge equal horizontal runs into vertically extended rectangles."""
    active: dict[tuple[int, int], tuple[int, int]] = {}
    rectangles: list[tuple[int, int, int, int]] = []
    for row in range(mask.shape[0] + 1):
        runs: set[tuple[int, int]] = set()
        if row < mask.shape[0]:
            xs = np.flatnonzero(mask[row])
            if len(xs):
                splits = np.flatnonzero(np.diff(xs) > 1)
                starts = np.r_[0, splits + 1]
                ends = np.r_[splits, len(xs) - 1]
                runs = {(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends, strict=True)}
        for run in list(active):
            if run not in runs:
                first, last = active.pop(run)
                rectangles.append((run[0], run[1], first, last))
        for run in runs:
            active[run] = (active[run][0], row) if run in active else (row, row)
    return rectangles


def _box_arrays(rectangles, low, resolution, z_low, z_high, geometry_box):
    shape_types = []
    half_extents = []
    positions = []
    orientations = []
    for x0, x1, y0, y1 in rectangles:
        x_min = low[0] + (x0 - 0.5) * resolution
        x_max = low[0] + (x1 + 0.5) * resolution
        y_min = low[1] + (y0 - 0.5) * resolution
        y_max = low[1] + (y1 + 0.5) * resolution
        shape_types.append(geometry_box)
        half_extents.append([(x_max - x_min) / 2, (y_max - y_min) / 2, (z_high - z_low) / 2])
        positions.append([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_high + z_low) / 2])
        orientations.append([0, 0, 0, 1])
    return shape_types, half_extents, positions, orientations


def _base_body(p, cid: int, base_obj: Path, resolution: float = 0.00020) -> tuple[int, float]:
    """Create an extruded ring from the base's horizontal top triangles."""
    import cv2
    import trimesh

    mesh = trimesh.load_mesh(str(base_obj), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    low = vertices[:, :2].min(axis=0) - resolution
    high = vertices[:, :2].max(axis=0) + resolution
    width, height = np.ceil((high - low) / resolution).astype(int) + 1
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    pixel_xy = (vertices[:, :2] - low) / resolution
    top = float(vertices[:, 2].max())
    for face in faces:
        if np.all(np.abs(vertices[face, 2] - top) < 1e-8):
            cv2.fillConvexPoly(mask, np.rint(pixel_xy[face]).astype(np.int32), 1)
    rectangles = _mask_rectangles(mask)
    z_low = float(vertices[:, 2].min())
    arrays = _box_arrays(rectangles, low, resolution, z_low, top, p.GEOM_BOX)
    collision = p.createCollisionShapeArray(
        shapeTypes=arrays[0],
        halfExtents=arrays[1],
        collisionFramePositions=arrays[2],
        collisionFrameOrientations=arrays[3],
        physicsClientId=cid,
    )
    visual = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(base_obj),
        rgbaColor=[0.7, 0.7, 0.7, 1.0],
        physicsClientId=cid,
    )
    body = p.createMultiBody(0, collision, visual, [0, 0, 0], [0, 0, 0, 1], physicsClientId=cid)
    return body, top


def _peg_body(p, cid: int, peg_obj: Path, pose, resolution: float = 0.00020) -> tuple[int, float]:
    """Create an extruded silhouette from native box primitives.

    Convex mesh collision shapes carry an implicit margin too large for the
    sub-millimetre clearance. Thin box runs retain concavities without that
    margin and match the collision representation used by the Panda scene.
    """
    import cv2
    import trimesh

    mesh = trimesh.load_mesh(str(peg_obj), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    low = vertices[:, :2].min(axis=0) - resolution
    high = vertices[:, :2].max(axis=0) + resolution
    width, height = np.ceil((high - low) / resolution).astype(int) + 1
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    pixel_xy = (vertices[:, :2] - low) / resolution
    for face in faces:
        cv2.fillConvexPoly(mask, np.rint(pixel_xy[face]).astype(np.int32), 1)

    rectangles = _mask_rectangles(mask)
    z_low = float(vertices[:, 2].min())
    z_high = float(vertices[:, 2].max())
    arrays = _box_arrays(rectangles, low, resolution, z_low, z_high, p.GEOM_BOX)
    collision = p.createCollisionShapeArray(
        shapeTypes=arrays[0],
        halfExtents=arrays[1],
        collisionFramePositions=arrays[2],
        collisionFrameOrientations=arrays[3],
        physicsClientId=cid,
    )
    visual = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(peg_obj),
        rgbaColor=[0.9, 0.9, 0.9, 1.0],
        physicsClientId=cid,
    )
    body = p.createMultiBody(
        0.1,
        collision,
        visual,
        pose[:3],
        pose[3:],
        physicsClientId=cid,
    )
    return body, z_low


def simulate_pybullet_insertion(
    shape: str,
    pose_error,
    config: InsertionConfig,
    registry: AssetRegistry | None = None,
    capture_path: str | Path | None = None,
) -> PyBulletInsertionResult:
    """Descend a measured peg and reject blocking mesh contact."""
    import pybullet as p

    reg = registry or AssetRegistry()
    asset = reg.get(shape)
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, 0, physicsClientId=cid)
        p.setTimeStep(1 / 240, physicsClientId=cid)
        base, top = _base_body(p, cid, asset.base_obj)
        pose = np.asarray(pose_error, dtype=float)
        yaw = p.getQuaternionFromEuler([0, 0, math.radians(float(pose[2]))])
        peg, z_min = _peg_body(p, cid, asset.peg_obj, [pose[0], pose[1], 0.2, *yaw])
        start_origin = top + config.approach_clearance_mm / 1000 - z_min
        start = np.asarray([pose[0], pose[1], start_origin])
        p.resetBasePositionAndOrientation(peg, start, yaw, physicsClientId=cid)

        contacts = []
        depth = 0.0
        attempts = 0
        max_force = 0.0
        max_penetration = 0.0
        reason = "timeout"
        for attempt in range(1, int(config.max_descent_attempts) + 1):
            attempts = attempt
            proposed = min(float(config.target_depth_mm), depth + float(config.descent_increment_mm))
            target = start.copy()
            target[2] -= proposed / 1000
            p.resetBasePositionAndOrientation(peg, target, yaw, physicsClientId=cid)
            for _ in range(4):
                p.stepSimulation(physicsClientId=cid)
            current = np.asarray(p.getBasePositionAndOrientation(peg, physicsClientId=cid)[0])
            depth = max(0.0, (start_origin - current[2]) * 1000.0)
            current_contacts = p.getContactPoints(peg, base, physicsClientId=cid)

            # Zero-distance/zero-force boundary reports are not blocking. The
            # penetration threshold is below raster/facet uncertainty while a
            # deliberate 2 mm error remains reliably detected on every shape.
            blocking = [contact for contact in current_contacts if float(contact[8]) < -6e-5 or float(contact[9]) > 0.1]
            contacts.extend(blocking)
            max_force = max(max_force, max((float(c[9]) for c in current_contacts), default=0.0))
            max_penetration = max(
                max_penetration,
                max((-float(c[8]) * 1000 for c in current_contacts), default=0.0),
            )
            if blocking:
                reason = "rim_collision"
                break
            # A 0.01 mm residual covers Bullet settling, not insertion clearance.
            if depth + 0.01 >= float(config.target_depth_mm):
                reason = "success"
                break

        final = np.asarray(p.getBasePositionAndOrientation(peg, physicsClientId=cid)[0])
        if capture_path is not None:
            from PIL import Image

            view = p.computeViewMatrix([0.025, -0.045, 0.04], [0, 0, top], [0, 0, 1])
            projection = p.computeProjectionMatrixFOV(45.0, 4.0 / 3.0, 0.005, 0.25)
            width, height, rgba, _depth, _ids = p.getCameraImage(
                640,
                480,
                viewMatrix=view,
                projectionMatrix=projection,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=cid,
            )
            destination = Path(capture_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4), mode="RGBA").save(destination)
        drift = float(np.linalg.norm(final[:2] - start[:2]) * 1000)
        return PyBulletInsertionResult(
            success=reason == "success",
            reason=reason,
            measured_depth_mm=depth,
            target_depth_mm=float(config.target_depth_mm),
            attempts=attempts,
            contact_count=len(contacts),
            max_normal_force_n=max_force,
            max_penetration_mm=max_penetration,
            lateral_drift_mm=drift,
        )
    finally:
        p.disconnect(cid)
