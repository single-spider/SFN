import hashlib
import json

import numpy as np
import pytest
from sfn.data.augment import apply_domain_randomization
from sfn.data.collect import collect_npz
from sfn.data.splits import CROSS_SHAPE_FOLD_VERSION, CROSS_SHAPE_FOLDS, get_cross_shape_fold
from sfn.data.validate import validate_dataset


def _image():
    rgb = np.zeros((3, 40, 50), dtype=np.uint8)
    rgb[:, 10:30, 15:35] = 180
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[10:20, 15:35] = 1
    mask[20:30, 15:35] = 2
    return rgb, mask


def test_randomization_is_deterministic_recorded_and_preserves_mask_classes():
    rgb, mask = _image()
    a_rgb, a_mask, a_record = apply_domain_randomization(rgb, mask, "heavy", seed=77, return_record=True)
    b_rgb, b_mask, b_record = apply_domain_randomization(rgb, mask, "heavy", seed=77, return_record=True)
    assert np.array_equal(a_rgb, b_rgb)
    assert np.array_equal(a_mask, b_mask)
    assert a_record == b_record
    assert set(np.unique(a_mask)) <= {0, 1, 2}
    assert {"camera", "occlusion", "noise", "blur", "exposure"} <= a_record.keys()
    assert not np.shares_memory(a_rgb, rgb)


def test_none_is_identity_and_invalid_inputs_fail():
    rgb, mask = _image()
    out_rgb, out_mask, record = apply_domain_randomization(rgb, mask, seed=4, return_record=True)
    assert np.array_equal(out_rgb, rgb) and np.array_equal(out_mask, mask)
    assert record["level"] == "none"
    with pytest.raises(ValueError, match="level"):
        apply_domain_randomization(rgb, mask, "volcanic")


def test_schema_v2_records_each_sample_and_strict_validation(tmp_path):
    collect_npz(tmp_path, samples_per_shape=1, seed=23, randomization_level="light")
    manifest = validate_dataset(tmp_path)
    assert manifest["schema_version"] == 2
    assert manifest["randomization"]["level"] == "light"
    assert manifest["schema_revision"] == 1
    assert len(manifest["config_hash"]) == 64 and manifest["source_revision"]
    assert {"intrinsics", "extrinsics", "crop"} <= manifest["camera_config"].keys()
    assert manifest["modalities"]["depth"] == {"required": False, "present": False, "units": "m"}
    assert manifest["physics_parameters"]["dynamics_enabled"] is False
    chunk = tmp_path / manifest["chunks"][0]["path"]
    with np.load(chunk, allow_pickle=False) as arrays:
        records = [json.loads(x) for x in arrays["augmentation_json"]]
        assert len(records) == manifest["samples"]
        assert all(r["seed"] is not None and r["level"] == "light" for r in records)
        assert {"shape_family", "symmetry_order", "episode_id", "frame_id"} <= set(arrays.files)
        assert np.all(arrays["frame_id"] == 0)


def test_revision_zero_v2_manifest_remains_valid(tmp_path):
    collect_npz(tmp_path, samples_per_shape=1, seed=29)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    for key in (
        "schema_revision",
        "schema_id",
        "source_revision",
        "config_hash",
        "resolved_config",
        "shape_catalog",
        "physics_parameters",
        "split_definition_version",
        "modalities",
    ):
        manifest.pop(key)
    manifest["camera_config"] = {"crop_width": 250, "crop_height": 200}
    path.write_text(json.dumps(manifest))
    assert validate_dataset(tmp_path)["schema_version"] == 2


def test_optional_metric_depth_is_declared_and_validated(tmp_path):
    collect_npz(tmp_path, samples_per_shape=1, seed=31)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for chunk in manifest["chunks"]:
        path = tmp_path / chunk["path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        arrays["depth"] = np.full(arrays["mask"].shape, 0.25, dtype=np.float32)
        np.savez_compressed(path, **arrays)
        chunk["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["modalities"]["depth"]["present"] = True
    manifest_path.write_text(json.dumps(manifest))
    assert validate_dataset(tmp_path)["modalities"]["depth"]["units"] == "m"


def test_versioned_cross_shape_folds_cover_each_shape_once():
    assert CROSS_SHAPE_FOLD_VERSION == "cross-shape-4fold-v1"
    held_out = [shape for fold in CROSS_SHAPE_FOLDS.values() for shape in fold["test"]]
    assert len(held_out) == len(set(held_out)) == 16
    for fold in range(4):
        train, test = set(get_cross_shape_fold(fold, "train")), set(get_cross_shape_fold(fold))
        assert train.isdisjoint(test) and train | test == set(held_out)


def test_strict_validation_rejects_manifest_version(tmp_path):
    collect_npz(tmp_path, samples_per_shape=1, seed=2)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema_version"):
        validate_dataset(tmp_path)
