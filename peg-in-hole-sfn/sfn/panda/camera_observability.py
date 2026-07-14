"""Native Panda-camera observability and body-ID mask validation tools.

The sweep renderer intentionally lives outside :mod:`panda_scene`.  It uses a
loaded ``PandaScene`` connection, but supplies its own view/projection matrices
so camera candidates can be evaluated without changing the runtime camera.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import MASK_BACKGROUND, MASK_PEG, MASK_SEAM
from .config import PandaConfig
from .panda_scene import PandaScene


@dataclass(frozen=True)
class CameraCandidate:
    """A complete native camera candidate, expressed in the task-origin frame."""

    eye_offset_m: tuple[float, float, float]
    target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov_y_deg: float = 45.0
    width: int = 250
    height: int = 200
    near: float = 0.001
    far: float = 1.0
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    name: str = ""

    @property
    def candidate_id(self) -> str:
        if self.name:
            return self.name
        eye = ",".join(f"{v:.3f}" for v in self.eye_offset_m)
        target = ",".join(f"{v:.3f}" for v in self.target_offset_m)
        return f"eye={eye}|target={target}|fov={self.fov_y_deg:g}|{self.width}x{self.height}"

    def validate(self) -> None:
        if self.width <= 1 or self.height <= 1:
            raise ValueError("camera width and height must exceed one pixel")
        if not 1.0 <= self.fov_y_deg < 179.0:
            raise ValueError("fov_y_deg must be in [1, 179)")
        if not 0.0 < self.near < self.far:
            raise ValueError("camera clipping planes must satisfy 0 < near < far")
        eye = np.asarray(self.eye_offset_m, dtype=np.float64)
        target = np.asarray(self.target_offset_m, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        if not np.isfinite(np.r_[eye, target, up]).all():
            raise ValueError("camera vectors must be finite")
        look = target - eye
        if np.linalg.norm(look) <= 1e-9 or np.linalg.norm(np.cross(look, up)) <= 1e-9:
            raise ValueError("camera eye/target/up vectors are degenerate")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_id"] = self.candidate_id
        return result


@dataclass(frozen=True)
class MaskValidation:
    valid: bool
    errors: tuple[str, ...]
    peg_pixels: int
    seam_pixels: int
    peg_clipped: bool
    seam_clipped: bool
    peg_centroid_xy: tuple[float, float] | None
    seam_centroid_xy: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SweepThresholds:
    min_peg_pixels: int = 20
    min_seam_pixels: int = 5
    min_visible_fraction: float = 0.95
    min_unclipped_fraction: float = 0.95
    min_xy_sensitivity_px_per_mm: float = 0.20
    min_yaw_change_per_deg: float = 0.0005


@dataclass
class CandidateAccumulator:
    candidate: CameraCandidate
    frames: list[dict[str, Any]] = field(default_factory=list)
    masks: dict[tuple[str, float, float, float], np.ndarray] = field(default_factory=dict, repr=False)


def decode_native_segmentation(
    segmentation: np.ndarray,
    peg_body_id: int,
    seam_body_id: int,
) -> np.ndarray:
    """Convert PyBullet packed body/link IDs to the project's three labels."""
    seg = np.asarray(segmentation, dtype=np.int64)
    object_ids = seg & ((1 << 24) - 1)
    mask = np.full(seg.shape, MASK_BACKGROUND, dtype=np.uint8)
    mask[object_ids == int(seam_body_id)] = MASK_SEAM
    # Peg wins if IDs are accidentally identical; callers validate scene IDs.
    mask[object_ids == int(peg_body_id)] = MASK_PEG
    return mask


def _centroid(binary: np.ndarray) -> tuple[float, float] | None:
    yy, xx = np.nonzero(binary)
    if len(xx) == 0:
        return None
    return float(xx.mean()), float(yy.mean())


def _touches_border(binary: np.ndarray) -> bool:
    return bool(
        binary.size and (binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any())
    )


def validate_native_mask(
    mask: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
    min_peg_pixels: int = 1,
    min_seam_pixels: int = 1,
) -> MaskValidation:
    """Validate labels, visibility, and border clipping in one native mask."""
    arr = np.asarray(mask)
    errors: list[str] = []
    if arr.ndim != 2:
        errors.append(f"mask must be 2-D, got shape {arr.shape}")
        return MaskValidation(False, tuple(errors), 0, 0, False, False, None, None)
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        errors.append(f"mask shape {arr.shape} does not match expected {expected_shape}")
    allowed = {MASK_BACKGROUND, MASK_PEG, MASK_SEAM}
    raw_labels = np.unique(arr)
    unexpected = [float(v) for v in raw_labels if not np.isfinite(v) or float(v) not in allowed]
    if unexpected:
        errors.append(f"unexpected labels: {unexpected}")
    peg = arr == MASK_PEG
    seam = arr == MASK_SEAM
    peg_pixels = int(peg.sum())
    seam_pixels = int(seam.sum())
    if peg_pixels < int(min_peg_pixels):
        errors.append(f"peg pixels {peg_pixels} below minimum {min_peg_pixels}")
    if seam_pixels < int(min_seam_pixels):
        errors.append(f"seam pixels {seam_pixels} below minimum {min_seam_pixels}")
    return MaskValidation(
        not errors,
        tuple(errors),
        peg_pixels,
        seam_pixels,
        _touches_border(peg),
        _touches_border(seam),
        _centroid(peg),
        _centroid(seam),
    )


def render_candidate_mask(scene: PandaScene, candidate: CameraCandidate) -> np.ndarray:
    """Render one body-ID mask with candidate matrices on an existing scene."""
    candidate.validate()
    ids = scene._require_ids()  # observability tooling is deliberately scene-adjacent
    if ids.peg == ids.seam:
        raise RuntimeError("peg and seam must have distinct PyBullet body IDs")
    p = scene.p
    origin = np.asarray(scene.task_transform.origin_world, dtype=np.float64)
    eye = origin + np.asarray(candidate.eye_offset_m, dtype=np.float64)
    target = origin + np.asarray(candidate.target_offset_m, dtype=np.float64)
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), list(candidate.up))
    projection = p.computeProjectionMatrixFOV(
        fov=float(candidate.fov_y_deg),
        aspect=float(candidate.width) / float(candidate.height),
        nearVal=float(candidate.near),
        farVal=float(candidate.far),
    )
    _w, _h, _rgba, _depth, segmentation = p.getCameraImage(
        width=int(candidate.width),
        height=int(candidate.height),
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=scene.client_id,
    )
    seg = np.asarray(segmentation, dtype=np.int64).reshape(candidate.height, candidate.width)
    return decode_native_segmentation(seg, ids.peg, ids.seam)


def canonical_angle_deg(angle_deg: float, period_deg: float) -> float:
    """Map an angle to the symmetry-aware interval [-period/2, period/2)."""
    if not np.isfinite(period_deg) or period_deg <= 0.0:
        raise ValueError("period_deg must be finite and positive")
    return float((float(angle_deg) + period_deg / 2.0) % period_deg - period_deg / 2.0)


def symmetry_aware_yaw_distance_deg(a_deg: float, b_deg: float, period_deg: float) -> float:
    return abs(canonical_angle_deg(float(a_deg) - float(b_deg), period_deg))


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    union = int(np.logical_or(aa, bb).sum())
    return 1.0 if union == 0 else float(np.logical_and(aa, bb).sum() / union)


def _center_binary(binary: np.ndarray) -> np.ndarray:
    arr = np.asarray(binary, dtype=np.uint8)
    center = _centroid(arr)
    if center is None:
        return arr.astype(bool)
    target_x = (arr.shape[1] - 1) / 2.0
    target_y = (arr.shape[0] - 1) / 2.0
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
        raise RuntimeError("opencv-python is required for mask alignment") from exc
    matrix = np.asarray([[1.0, 0.0, target_x - center[0]], [0.0, 1.0, target_y - center[1]]])
    shifted = cv2.warpAffine(arr, matrix, (arr.shape[1], arr.shape[0]), flags=cv2.INTER_NEAREST)
    return shifted.astype(bool)


def infer_rotational_symmetry_order(
    silhouette: np.ndarray,
    *,
    max_order: int = 12,
    iou_threshold: float = 0.97,
) -> tuple[int, dict[int, float]]:
    """Infer top-view rotational symmetry using centered silhouette overlap.

    The highest passing order is returned.  Scores are included so reports do
    not silently turn a noisy mesh/raster decision into supposed ground truth.
    """
    binary = _center_binary(np.asarray(silhouette, dtype=bool))
    if int(binary.sum()) == 0:
        return 1, {}
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for symmetry inference") from exc
    center = ((binary.shape[1] - 1) / 2.0, (binary.shape[0] - 1) / 2.0)
    scores: dict[int, float] = {}
    passing = [1]
    source = binary.astype(np.uint8)
    for order in range(2, int(max_order) + 1):
        matrix = cv2.getRotationMatrix2D(center, 360.0 / order, 1.0)
        rotated = cv2.warpAffine(source, matrix, (source.shape[1], source.shape[0]), flags=cv2.INTER_NEAREST)
        score = _binary_iou(binary, rotated > 0)
        scores[order] = score
        if score >= float(iou_threshold):
            passing.append(order)
    return max(passing), scores


def load_peg_top_silhouette(
    shape: str,
    asset_root: str | Path,
    *,
    image_size: int = 256,
    padding_fraction: float = 0.08,
) -> np.ndarray:
    """Rasterize the actual peg OBJ's top silhouette for symmetry diagnosis."""
    try:
        import cv2
        import trimesh
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("trimesh and opencv-python are required for symmetry diagnosis") from exc
    path = Path(asset_root) / shape / "peg" / "peg.obj"
    mesh = trimesh.load_mesh(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[:, :2]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    low = vertices.min(axis=0)
    extent = np.maximum(vertices.max(axis=0) - low, 1e-12)
    usable = float(image_size) * (1.0 - 2.0 * float(padding_fraction))
    scale = usable / float(extent.max())
    pixels = (vertices - low) * scale
    occupied = extent * scale
    pixels += (float(image_size) - occupied) / 2.0
    canvas = np.zeros((int(image_size), int(image_size)), dtype=np.uint8)
    for face in faces:
        triangle = np.rint(pixels[face]).astype(np.int32)
        cv2.fillConvexPoly(canvas, triangle, 1, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def _linear_xy_sensitivity(frames: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    usable = [f for f in frames if f["peg_centroid_xy"] is not None]
    if len(usable) < 3:
        return {"x_px_per_mm": None, "y_px_per_mm": None, "minimum_px_per_mm": None}
    poses = np.asarray([[f["pose_error"][0] * 1000.0, f["pose_error"][1] * 1000.0, 1.0] for f in usable])
    centroids = np.asarray([f["peg_centroid_xy"] for f in usable], dtype=np.float64)
    if np.linalg.matrix_rank(poses) < 2:
        return {"x_px_per_mm": None, "y_px_per_mm": None, "minimum_px_per_mm": None}
    weights = np.linalg.lstsq(poses, centroids, rcond=None)[0][:2, :]
    singular = np.linalg.svd(weights, compute_uv=False)
    return {
        "x_px_per_mm": float(np.linalg.norm(weights[0])),
        "y_px_per_mm": float(np.linalg.norm(weights[1])),
        "minimum_px_per_mm": float(np.min(singular)),
    }


def yaw_observability_diagnostic(
    masks_by_pose: dict[tuple[float, float, float], np.ndarray],
    symmetry_order: int,
    *,
    ambiguity_iou: float = 0.995,
) -> dict[str, Any]:
    """Compare centered peg silhouettes at equal XY and distinct yaw values."""
    period = 360.0 / max(1, int(symmetry_order))
    groups: dict[tuple[float, float], list[tuple[float, np.ndarray]]] = {}
    for (x_m, y_m, yaw_deg), mask in masks_by_pose.items():
        peg = np.asarray(mask) == MASK_PEG
        if peg.any():
            groups.setdefault((x_m, y_m), []).append((yaw_deg, _center_binary(peg)))
    rows: list[dict[str, float]] = []
    ambiguity_pairs = 0
    for values in groups.values():
        for (yaw_a, mask_a), (yaw_b, mask_b) in combinations(values, 2):
            delta = symmetry_aware_yaw_distance_deg(yaw_a, yaw_b, period)
            if delta <= 1e-9:
                continue
            iou = _binary_iou(mask_a, mask_b)
            change_per_deg = (1.0 - iou) / delta
            rows.append({"canonical_delta_deg": delta, "iou": iou, "change_per_deg": change_per_deg})
            ambiguity_pairs += int(iou >= ambiguity_iou)
    changes = np.asarray([row["change_per_deg"] for row in rows], dtype=np.float64)
    return {
        "symmetry_order": int(symmetry_order),
        "symmetry_period_deg": period,
        "pair_count": len(rows),
        "ambiguity_pairs": ambiguity_pairs,
        "median_change_per_deg": None if not len(changes) else float(np.median(changes)),
        "min_change_per_deg": None if not len(changes) else float(np.min(changes)),
        "max_change_per_deg": None if not len(changes) else float(np.max(changes)),
    }


def summarize_candidate(
    accumulator: CandidateAccumulator,
    symmetry_by_shape: dict[str, int],
    thresholds: SweepThresholds,
) -> dict[str, Any]:
    frames = accumulator.frames
    total = len(frames)
    visible = [
        f
        for f in frames
        if f["peg_pixels"] >= thresholds.min_peg_pixels and f["seam_pixels"] >= thresholds.min_seam_pixels
    ]
    unclipped = [f for f in frames if not f["peg_clipped"] and not f["seam_clipped"]]
    visible_fraction = float(len(visible) / total) if total else 0.0
    unclipped_fraction = float(len(unclipped) / total) if total else 0.0
    sensitivity = _linear_xy_sensitivity(frames)
    yaw_by_shape: dict[str, Any] = {}
    for shape in sorted({f["shape"] for f in frames}):
        shape_masks = {
            (x, y, yaw): mask for (mask_shape, x, y, yaw), mask in accumulator.masks.items() if mask_shape == shape
        }
        yaw_by_shape[shape] = yaw_observability_diagnostic(shape_masks, symmetry_by_shape.get(shape, 1))
    yaw_values = [d["median_change_per_deg"] for d in yaw_by_shape.values() if d["median_change_per_deg"] is not None]
    min_xy = sensitivity["minimum_px_per_mm"]
    yaw_floor = min(yaw_values) if yaw_values else None
    reasons: list[str] = []
    if visible_fraction < thresholds.min_visible_fraction:
        reasons.append("visibility")
    if unclipped_fraction < thresholds.min_unclipped_fraction:
        reasons.append("clipping")
    if min_xy is None or min_xy < thresholds.min_xy_sensitivity_px_per_mm:
        reasons.append("xy_pixel_sensitivity")
    if yaw_floor is None or yaw_floor < thresholds.min_yaw_change_per_deg:
        reasons.append("yaw_pixel_sensitivity")
    return {
        "candidate": accumulator.candidate.to_dict(),
        "viable": not reasons,
        "rejection_reasons": reasons,
        "frame_count": total,
        "visible_fraction": visible_fraction,
        "unclipped_fraction": unclipped_fraction,
        "min_peg_pixels": min((f["peg_pixels"] for f in frames), default=0),
        "min_seam_pixels": min((f["seam_pixels"] for f in frames), default=0),
        "xy_sensitivity": sensitivity,
        "yaw_diagnostics": yaw_by_shape,
        "frames": frames,
    }


def sweep_camera_observability(
    *,
    shapes: Sequence[str],
    poses: Iterable[Sequence[float]],
    candidates: Sequence[CameraCandidate],
    panda_config: PandaConfig | None = None,
    asset_root: str | Path | None = None,
    thresholds: SweepThresholds | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Sweep actual Panda scenes over shape, pose, and camera grids."""
    if not shapes:
        raise ValueError("at least one shape is required")
    if not candidates:
        raise ValueError("at least one camera candidate is required")
    pose_rows = [tuple(float(v) for v in np.asarray(pose).reshape(3)) for pose in poses]
    if not pose_rows:
        raise ValueError("at least one pose is required")
    for candidate in candidates:
        candidate.validate()
    limits = thresholds or SweepThresholds()
    accumulators = {candidate.candidate_id: CandidateAccumulator(candidate) for candidate in candidates}
    symmetry_by_shape: dict[str, int] = {}
    symmetry_scores: dict[str, dict[int, float]] = {}
    scene_config = panda_config or PandaConfig(native_camera=True, mesh_derived_alignment_z=True)

    for shape_index, shape in enumerate(shapes):
        with PandaScene(
            shape=shape,
            config=scene_config,
            asset_root=asset_root,
            seed=int(seed) + shape_index,
        ) as scene:
            silhouette = load_peg_top_silhouette(shape, scene.asset_root)
            order, scores = infer_rotational_symmetry_order(silhouette)
            symmetry_by_shape[shape] = order
            symmetry_scores[shape] = scores
            for pose in pose_rows:
                scene.reset_to_pose_error(pose)
                measured = scene.measure().pose_error_task.tolist()
                for candidate in candidates:
                    mask = render_candidate_mask(scene, candidate)
                    validation = validate_native_mask(
                        mask,
                        expected_shape=(candidate.height, candidate.width),
                        min_peg_pixels=limits.min_peg_pixels,
                        min_seam_pixels=limits.min_seam_pixels,
                    )
                    accumulator = accumulators[candidate.candidate_id]
                    accumulator.frames.append(
                        {
                            "shape": shape,
                            "pose_error": list(pose),
                            "measured_pose_error": measured,
                            **validation.to_dict(),
                        }
                    )
                    accumulator.masks[(shape, pose[0], pose[1], pose[2])] = mask

    summaries = [summarize_candidate(acc, symmetry_by_shape, limits) for acc in accumulators.values()]
    summaries.sort(
        key=lambda row: (
            not row["viable"],
            -row["visible_fraction"],
            -row["unclipped_fraction"],
            -(row["xy_sensitivity"]["minimum_px_per_mm"] or -1.0),
        )
    )
    return {
        "schema_version": 1,
        "renderer": "panda_native_pybullet_body_id",
        "shapes": list(shapes),
        "poses": [list(pose) for pose in pose_rows],
        "thresholds": asdict(limits),
        "symmetry": {
            shape: {
                "order": symmetry_by_shape[shape],
                "period_deg": 360.0 / symmetry_by_shape[shape],
                "rotation_iou_by_order": symmetry_scores[shape],
            }
            for shape in shapes
        },
        "viable_candidate_ids": [row["candidate"]["candidate_id"] for row in summaries if row["viable"]],
        "recommended_candidate_id": next(
            (row["candidate"]["candidate_id"] for row in summaries if row["viable"]), None
        ),
        "candidates": summaries,
        "core_integration": {
            "required": False,
            "runtime_camera_switching_requires_core_change": True,
            "reason": (
                "PandaScene.render_camera reads eye/target/up from PandaConfig and resolution/FOV from "
                "CameraConfig. Apply a selected candidate to those configs at construction time; no core "
                "change is needed for static deployment. Runtime camera switching would require a public "
                "camera-override argument or renderer object in PandaScene."
            ),
        },
    }
