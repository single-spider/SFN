from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from scripts.create_source_snapshot import create_outputs
from scripts.stamp_release_checkpoints import stamp_release, state_dict_sha256
from sfn.training.common import load_checkpoint_cpu


def test_source_snapshot_and_archive_are_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "sfn").mkdir(parents=True)
    (root / "sfn" / "module.py").write_bytes(b"VALUE = 1\n")
    (root / "sfn" / "__pycache__").mkdir()
    (root / "sfn" / "__pycache__" / "module.pyc").write_bytes(b"cache")
    first_manifest = tmp_path / "first.json"
    second_manifest = tmp_path / "second.json"
    first_archive = tmp_path / "first.zip"
    second_archive = tmp_path / "second.zip"

    first = create_outputs(root, first_manifest, first_archive)
    second = create_outputs(root, second_manifest, second_archive)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first["archive_sha256"] == second["archive_sha256"]
    assert json.loads(first_manifest.read_text())["files"] == [
        {
            "bytes": 10,
            "path": "sfn/module.py",
            "sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        }
    ]


def test_release_stamping_preserves_weights_and_records_known_metadata(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "models").mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "data" / "mesh_v2_train").mkdir(parents=True)
    config = root / "configs" / "segmentation.yaml"
    config.write_text("task: segmentation\n", encoding="utf-8")
    chunk = root / "data" / "mesh_v2_train" / "train.npz"
    chunk.write_bytes(b"samples")
    chunk_sha = hashlib.sha256(b"samples").hexdigest()
    split = {"chunks": [{"path": chunk.name, "samples": 1, "sha256": chunk_sha}]}
    (chunk.parent / "manifest.json").write_text(json.dumps(split), encoding="utf-8")
    checkpoint_path = root / "models" / "selected.pt"
    torch.save(
        {
            "schema_version": 1,
            "model_name": "Tiny",
            "model_config": {"width": 2},
            "model_state_dict": {"weight": torch.arange(4).reshape(2, 2), "counter": torch.tensor(1)},
            "epoch": 3,
            "metrics": {
                "selection_metric": "mean_iou",
                "selection_mode": "max",
                "selection_value": 0.75,
                "val": {"mean_iou": 0.75, "matrix": [[1]]},
            },
            "data_split": split,
            "train_config": {"epochs": 3},
        },
        checkpoint_path,
    )
    before = state_dict_sha256(load_checkpoint_cpu(checkpoint_path)["model_state_dict"])
    release_manifest = root / "configs" / "release_checkpoints.json"
    release_manifest.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "component": "segmentation",
                        "path": "models/selected.pt",
                        "config_paths": ["configs/segmentation.yaml"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    snapshot = root / "snapshot.json"
    snapshot.write_text(
        json.dumps({"aggregate_sha256": "abc", "file_count": 1, "git_head": None}), encoding="utf-8"
    )

    first = stamp_release(root, release_manifest, snapshot)
    second = stamp_release(root, release_manifest, snapshot)
    stamped = load_checkpoint_cpu(checkpoint_path)
    release = stamped["provenance"]["release"]

    assert first[0]["action"] == "stamped"
    assert second[0]["action"] == "unchanged"
    assert state_dict_sha256(stamped["model_state_dict"]) == before == release["weights_sha256"]
    assert release["dataset"]["status"] == "known"
    assert release["selection"]["value"] == 0.75
    assert release["selection"]["validation_metrics"] == {"mean_iou": 0.75}
