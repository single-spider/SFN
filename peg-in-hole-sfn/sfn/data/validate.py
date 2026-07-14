from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .augment import LEVELS
from .schema import (
    OPTIONAL_ARRAYS_V2,
    REQUIRED_ARRAYS_V2_CORE,
    REQUIRED_ARRAYS_V2_REV1,
    SCHEMA_REVISION,
    SCHEMA_VERSION,
    canonical_config_hash,
    shape_family,
    symmetry_order,
)


def validate_mask_classes(mask: np.ndarray) -> None:
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1, 2}):
        raise ValueError(f"Mask values must be subset of {{0,1,2}}, got {sorted(values)}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_dataset(root: str | Path) -> dict:
    """Validate schema-v2, including legacy v2 and the provenance-rich revision 1."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        f"Unsupported dataset schema_version; expected {SCHEMA_VERSION}",
    )
    revision = int(manifest.get("schema_revision", 0))
    _require(0 <= revision <= SCHEMA_REVISION, f"Unsupported schema_revision {revision}")
    for key in ("samples", "split", "chunks", "seed", "shapes", "randomization", "camera_config"):
        _require(key in manifest, f"Manifest missing required field: {key}")
    _require(manifest.get("metadata_path") == "manifest.json", "metadata_path must be 'manifest.json'")
    _require(isinstance(manifest["chunks"], list) and manifest["chunks"], "Manifest chunks must be a non-empty list")
    randomization = manifest["randomization"]
    _require(
        isinstance(randomization, dict) and randomization.get("level") in LEVELS,
        "Invalid manifest randomization configuration",
    )
    if revision >= 1:
        for key in (
            "schema_id",
            "source_revision",
            "config_hash",
            "resolved_config",
            "shape_catalog",
            "physics_parameters",
            "split_definition_version",
            "modalities",
        ):
            _require(key in manifest, f"Manifest missing schema-v2 revision-1 field: {key}")
        _require(
            manifest["config_hash"] == canonical_config_hash(manifest["resolved_config"]),
            "Manifest config_hash does not match resolved_config",
        )
        camera = manifest["camera_config"]
        _require(isinstance(camera, dict), "camera_config must be an object")
        for key in ("intrinsics", "extrinsics", "crop"):
            _require(isinstance(camera.get(key), dict), f"camera_config missing {key}")
        _require(isinstance(manifest["physics_parameters"], dict), "physics_parameters must be an object")
        depth_decl = manifest["modalities"].get("depth")
        _require(isinstance(depth_decl, dict) and depth_decl.get("required") is False, "depth must be optional")

    total = 0
    seen_ids: set[int] = set()
    for chunk in manifest["chunks"]:
        _require(isinstance(chunk, dict), "Schema-v2 chunk entries must be objects")
        _require(set(("path", "sha256", "samples")).issubset(chunk), "Chunk entry missing path/sha256/samples")
        path = root / chunk["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset chunk: {path}")
        _require(_sha256(path) == chunk["sha256"], f"Checksum mismatch for {path}")
        with np.load(path, allow_pickle=False) as arrays:
            required_arrays = REQUIRED_ARRAYS_V2_REV1 if revision >= 1 else REQUIRED_ARRAYS_V2_CORE
            missing = required_arrays.difference(arrays.files)
            _require(not missing, f"{path} missing arrays: {sorted(missing)}")
            n = int(arrays["rgb"].shape[0])
            _require(n > 0 and int(chunk["samples"]) == n, f"Chunk sample count mismatch for {path}")
            for name in required_arrays:
                _require(arrays[name].shape[0] == n, f"{name} first dimension mismatch in {path}")
            crop = manifest["camera_config"]
            expected_h = int(crop.get("crop_height", crop.get("crop", {}).get("height", 200)))
            expected_w = int(crop.get("crop_width", crop.get("crop", {}).get("width", 250)))
            _require(
                arrays["rgb"].shape == (n, 3, expected_h, expected_w) and arrays["rgb"].dtype == np.uint8,
                f"Bad rgb shape/dtype {arrays['rgb'].shape}/{arrays['rgb'].dtype}",
            )
            _require(
                arrays["mask"].shape == (n, expected_h, expected_w)
                and np.issubdtype(arrays["mask"].dtype, np.integer),
                f"Bad mask shape/dtype in {path}",
            )
            _require(arrays["pose_error"].shape == (n, 3), f"Bad pose_error shape {arrays['pose_error'].shape}")
            _require(
                arrays["position_target"].shape == (n, 21, 21),
                f"Bad position_target shape {arrays['position_target'].shape}",
            )
            _require(arrays["orientation_index"].shape == (n,), f"Bad orientation_index shape in {path}")
            validate_mask_classes(arrays["mask"])
            validate_mask_classes(arrays["position_target"])
            _require(np.all(np.isfinite(arrays["pose_error"])), f"Non-finite pose_error in {path}")
            _require(
                not np.any((arrays["orientation_index"] < 0) | (arrays["orientation_index"] > 10)),
                f"orientation_index values out of range in {path}",
            )
            if "depth" in arrays.files:
                depth = arrays["depth"]
                _require(depth.shape == (n, expected_h, expected_w), f"Bad depth shape {depth.shape} in {path}")
                _require(np.issubdtype(depth.dtype, np.floating), f"depth must use a floating dtype in {path}")
                _require(np.all(np.isfinite(depth)) and np.all(depth >= 0), f"Invalid metric depth in {path}")
            unknown = set(arrays.files).difference(required_arrays | OPTIONAL_ARRAYS_V2)
            # Additive application-specific arrays remain legal for backwards
            # compatibility; the set is computed to make that policy explicit.
            del unknown
            if revision >= 1:
                for sid, family, order in zip(
                    arrays["shape_id"], arrays["shape_family"], arrays["symmetry_order"], strict=True
                ):
                    _require(str(family) == shape_family(str(sid)), f"shape_family mismatch for {sid!r} in {path}")
                    _require(int(order) == symmetry_order(str(sid)), f"symmetry_order mismatch for {sid!r} in {path}")
                _require(np.all(arrays["frame_id"] >= 0), f"frame_id must be non-negative in {path}")
            for raw, sample_id in zip(arrays["augmentation_json"], arrays["sample_id"], strict=True):
                try:
                    record = json.loads(str(raw))
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid augmentation_json in {path}: {exc}") from exc
                _require(
                    record.get("level") == randomization["level"] and record.get("version") == 1,
                    f"Augmentation record does not match manifest in {path}",
                )
                sid = int(sample_id)
                _require(sid not in seen_ids, f"Duplicate sample_id {sid}")
                seen_ids.add(sid)
            total += n
    _require(int(manifest["samples"]) == total, f"Manifest sample count {manifest['samples']} != chunk count {total}")
    _require(seen_ids == set(range(total)), "sample_id values must be contiguous from zero")
    if revision >= 1:
        declared_depth = bool(manifest["modalities"]["depth"].get("present"))
        with_depth = []
        for chunk in manifest["chunks"]:
            with np.load(root / chunk["path"], allow_pickle=False) as arrays:
                with_depth.append("depth" in arrays.files)
        _require(all(with_depth) == declared_depth and len(set(with_depth)) == 1, "depth modality declaration mismatch")
    return manifest
