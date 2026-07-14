import pytest
from sfn.constants import TRAIN_SEEN_SHAPES
from sfn.data.collect import collect_npz
from sfn.data.dataset import NPZDataset
from sfn.data.validate import validate_dataset


def test_collect_manifest_validate_and_load(tmp_path):
    chunk = collect_npz(tmp_path, split="train_seen", samples_per_shape=1, seed=12)
    assert chunk.exists()
    manifest = validate_dataset(tmp_path)
    assert manifest["samples"] == len(TRAIN_SEEN_SHAPES)
    assert manifest["chunks"][0]["sha256"]
    ds = NPZDataset(tmp_path)
    assert len(ds) == len(TRAIN_SEEN_SHAPES)
    sample = ds[0]
    assert sample["rgb"].shape == (3, 200, 250)
    assert sample["mask"].shape == (200, 250)
    assert sample["position_target"].shape == (21, 21)


def test_chunked_dataset_loads_all_chunks_and_edge_cases(tmp_path):
    collect_npz(tmp_path, split="train_seen", samples_per_shape=1, seed=13, chunk_size=3, include_edge_cases=True)
    manifest = validate_dataset(tmp_path)
    assert len(manifest["chunks"]) > 1
    ds = NPZDataset(tmp_path)
    assert len(ds) == manifest["samples"]
    assert ds[0]["rgb"].shape == (3, 200, 250)
    assert ds[len(ds) - 1]["mask"].shape == (200, 250)


def test_missing_dataset_has_actionable_error(tmp_path):
    missing = tmp_path / "missing_val_dataset"
    with pytest.raises(FileNotFoundError, match="Collect it first"):
        NPZDataset(missing)
