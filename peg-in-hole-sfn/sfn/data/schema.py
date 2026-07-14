from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetSample:
    rgb: np.ndarray
    mask: np.ndarray
    pose_error: np.ndarray
    position_target: np.ndarray
    orientation_index: int
    orientation_angle_deg: float
    shape_id: str
    sample_id: int
    seed: int
    camera_variant: int = 0
    domain_randomization: dict | None = None
    # Additive v2 fields are optional here so callers constructing the original
    # DatasetSample continue to work. Collectors should populate IDs/family.
    depth: np.ndarray | None = None
    shape_family: str | None = None
    symmetry_order: int = 1
    episode_id: int | str | None = None
    frame_id: int | None = None


SCHEMA_VERSION = 2
SCHEMA_REVISION = 1
SCHEMA_ID = "sfn.dataset/v2"
SPLIT_DEFINITION_VERSION = "shape-disjoint-v1"

# Arrays required by every v2 dataset, including datasets written before the
# additive contract extension. Do not grow this set without a schema bump.
REQUIRED_ARRAYS_V2 = frozenset(
    {
        "rgb",
        "mask",
        "pose_error",
        "position_target",
        "orientation_index",
        "shape_id",
        "sample_id",
        "seed",
        "camera_variant",
        "augmentation_json",
    }
)

# Extended fields are capability-based: new collectors emit all except depth;
# old v2 chunks remain valid. Depth, when present, is required to align one-for-
# one with RGB frames and is declared in the manifest's array contract.
OPTIONAL_ARRAYS_V2 = frozenset({"depth", "shape_family", "symmetry_order", "episode_id", "frame_id"})
NEW_COLLECTOR_ARRAYS_V2 = frozenset(OPTIONAL_ARRAYS_V2 - {"depth"})
# Explicit aliases make the compatibility boundary available to validators:
# revision 0 needs only the historical core; revision 1 writers add identity
# and geometry fields while depth remains optional.
REQUIRED_ARRAYS_V2_CORE = REQUIRED_ARRAYS_V2
REQUIRED_ARRAYS_V2_REV1 = REQUIRED_ARRAYS_V2 | NEW_COLLECTOR_ARRAYS_V2
# Revision names make the backwards-compatibility boundary explicit.
REQUIRED_ARRAYS_V2_CORE = REQUIRED_ARRAYS_V2
REQUIRED_ARRAYS_V2_REV1 = REQUIRED_ARRAYS_V2_CORE | NEW_COLLECTOR_ARRAYS_V2


_SYMMETRY_ORDER = {
    "triangle": 3,
    "square": 4,
    "diamond": 4,
    "pentagon": 5,
    "hexagon": 6,
}


def shape_family(shape_id: str) -> str:
    """Return a stable coarse family without depending on asset filenames."""
    leaf = str(shape_id).rsplit("-", 1)[-1]
    for prefix in ("concave", "convex", "fillet"):
        if leaf.startswith(prefix):
            return prefix
    return leaf


def shape_symmetry_order(shape_id: str) -> int:
    """Rotational symmetry order; unknown/custom geometry is conservatively 1."""
    return _SYMMETRY_ORDER.get(shape_family(shape_id), 1)


# Public spelling retained for collectors developed against the v2 proposal.
symmetry_order = shape_symmetry_order


def shape_catalog(shape_ids: list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        shape: {"family": shape_family(shape), "symmetry_order": shape_symmetry_order(shape)}
        for shape in shape_ids
    }


def canonical_hash(value: Any) -> str:
    """SHA-256 of JSON-normalized configuration/provenance data."""
    if is_dataclass(value):
        value = asdict(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


canonical_config_hash = canonical_hash


def source_revision(root: str | Path | None = None) -> str:
    """Best-effort immutable source revision, with an explicit unknown value."""
    override = os.environ.get("SFN_SOURCE_REVISION")
    if override:
        return override
    cwd = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def camera_contract(
    camera: Any,
    *,
    model: str | None = None,
    eye_m: tuple[float, float, float] | None = None,
    target_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Build a serializable intrinsics/extrinsics/crop contract."""
    cfg = asdict(camera) if is_dataclass(camera) else dict(camera)
    width, height = int(cfg["crop_width"]), int(cfg["crop_height"])
    render_width = int(cfg.get("render_width", width))
    render_height = int(cfg.get("render_height", height))
    backend = str(cfg.get("renderer_backend", "pinhole"))
    model = model or ("orthographic" if backend == "mesh_orthographic" else "pinhole")
    if model == "pinhole":
        fy = height / (2.0 * math.tan(math.radians(float(cfg["fov_y_deg"])) / 2.0))
        intrinsics = {
            "model": model,
            "matrix": [[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
            "near_m": float(cfg.get("near", 0.001)),
            "far_m": float(cfg.get("far", 10.0)),
        }
    else:
        scale = float(cfg.get("orthographic_pixels_per_mm", 1.0))
        intrinsics = {
            "model": model,
            "pixels_per_mm": scale,
            "principal_point_px": [width / 2.0, height / 2.0],
        }
    eye = tuple(float(v) for v in (eye_m or cfg.get("eye_offset", (0.0, 0.0, 0.0))))
    up_value = tuple(float(v) for v in (up or cfg.get("up", (0.0, -1.0, 0.0))))
    return {
        "intrinsics": intrinsics,
        "extrinsics": {
            "convention": "task_frame_eye_target_up",
            "eye_m": list(eye),
            "target_m": [float(v) for v in target_m],
            "up": list(up_value),
        },
        "crop": {
            "x_px": (render_width - width) // 2,
            "y_px": (render_height - height) // 2,
            "x": (render_width - width) // 2,
            "y": (render_height - height) // 2,
            "width": width,
            "height": height,
            "width_px": width,
            "height_px": height,
            "source_width_px": render_width,
            "source_height_px": render_height,
        },
    }


def array_contract(*, depth_present: bool = False) -> dict[str, Any]:
    present = sorted(REQUIRED_ARRAYS_V2 | NEW_COLLECTOR_ARRAYS_V2 | ({"depth"} if depth_present else set()))
    return {
        "required": sorted(REQUIRED_ARRAYS_V2),
        "optional": {
            "depth": {"present": bool(depth_present), "units": "m", "aligned_with": "rgb"},
            "shape_family": {"present": True},
            "symmetry_order": {"present": True, "meaning": "rotational order"},
            "episode_id": {"present": True},
            "frame_id": {"present": True},
        },
        "present": present,
    }
