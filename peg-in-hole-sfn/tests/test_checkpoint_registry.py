from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from scripts.audit.checkpoint_registry import build_registry, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"traceable")
    assert sha256_file(artifact) == hashlib.sha256(b"traceable").hexdigest()


def test_registry_links_checkpoint_to_verified_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    dataset_dir = data_dir / "mesh_v2_train"
    dataset_dir.mkdir(parents=True)
    models_dir.mkdir()
    chunk = dataset_dir / "train_000.npz"
    chunk.write_bytes(b"dataset")
    chunk_hash = sha256_file(chunk)
    manifest = {
        "schema_version": 2,
        "split": "train_seen",
        "samples": 1,
        "seed": 7,
        "renderer_backend": "mesh_orthographic",
        "chunks": [{"path": chunk.name, "sha256": chunk_hash, "samples": 1}],
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = {
        "schema_version": 1,
        "model_name": "PositionNet",
        "model_config": {"task": "position", "base": 8},
        "model_state_dict": {"weight": torch.ones(3)},
        "epoch": 2,
        "global_step": 5,
        "metrics": {
            "task": "position",
            "selection_metric": "mean_radial_error_mm",
            "selection_mode": "min",
            "selection_value": 0.5,
            "train": {"samples": 1, "loss": 0.2},
            "val": {"samples": 1, "mean_radial_error_mm": 0.5, "confusion": [[1]]},
        },
        "data_split": manifest,
    }
    torch.save(checkpoint, models_dir / "position_mesh_v2.pt")

    registry = build_registry(tmp_path, data_dir, models_dir, "*mesh_v2*.pt")

    assert len(registry["datasets"]) == 1
    assert registry["datasets"][0]["all_chunks_verified"] is True
    assert registry["checkpoints"][0]["parameter_count"] == 3
    assert registry["checkpoints"][0]["training_dataset"]["manifest_sha256"] == sha256_file(manifest_path)
    assert "confusion" not in registry["checkpoints"][0]["validation_metrics"]
