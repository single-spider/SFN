#!/usr/bin/env python
"""Generate a verifiable registry for checkpoints and their datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _dataset_content_sha256(chunks: list[dict[str, Any]]) -> str:
    """Hash the ordered manifest chunk identities, sizes, and declared digests."""
    records = [
        {
            "path": str(chunk.get("path", "")).replace("\\", "/"),
            "samples": chunk.get("samples"),
            "sha256": chunk.get("sha256"),
        }
        for chunk in chunks
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif key == "class_iou" and isinstance(value, dict):
            result[key] = value
    return result


def _dataset_record(manifest_path: Path, root: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = manifest.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"Manifest chunks must be a list: {manifest_path}")

    verified_chunks: list[dict[str, Any]] = []
    fingerprint: list[str] = []
    for item in chunks:
        chunk = dict(item)
        declared = chunk.get("sha256")
        fingerprint.append(str(declared))
        chunk_path = manifest_path.parent / str(chunk.get("path", ""))
        actual = sha256_file(chunk_path) if chunk_path.is_file() else None
        chunk["actual_sha256"] = actual
        chunk["hash_verified"] = actual == declared if actual is not None else False
        verified_chunks.append(chunk)

    record = {
        "name": manifest_path.parent.name,
        "manifest_path": _relative(manifest_path, root),
        "manifest_sha256": sha256_file(manifest_path),
        "content_sha256": _dataset_content_sha256(chunks),
        "schema_version": manifest.get("schema_version"),
        "split": manifest.get("split"),
        "samples": manifest.get("samples"),
        "seed": manifest.get("seed"),
        "shapes": manifest.get("shapes", []),
        "renderer_backend": manifest.get("renderer_backend"),
        "randomization": manifest.get("randomization"),
        "chunks": verified_chunks,
        "all_chunks_verified": bool(verified_chunks) and all(c["hash_verified"] for c in verified_chunks),
    }
    return record, tuple(fingerprint)


def _summary_path(checkpoint_path: Path) -> Path:
    name = checkpoint_path.name
    base = name.removesuffix(".last.pt") if name.endswith(".last.pt") else name.removesuffix(".pt")
    return checkpoint_path.with_name(f"{base}.summary.json")


def _checkpoint_record(
    checkpoint_path: Path,
    root: Path,
    dataset_by_fingerprint: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyTorch is required to read checkpoint metadata") from exc

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {checkpoint_path}")

    model_state = payload.get("model_state_dict", {})
    parameter_count = sum(int(value.numel()) for value in model_state.values() if hasattr(value, "numel"))
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    validation = metrics.get("val", {}) if isinstance(metrics.get("val"), dict) else {}
    training = metrics.get("train", {}) if isinstance(metrics.get("train"), dict) else {}
    data_split = payload.get("data_split", {}) if isinstance(payload.get("data_split"), dict) else {}
    embedded_chunks = data_split.get("chunks", []) if isinstance(data_split.get("chunks"), list) else []
    fingerprint = tuple(str(item.get("sha256")) for item in embedded_chunks if isinstance(item, dict))
    dataset = dataset_by_fingerprint.get(fingerprint)

    summary_path = _summary_path(checkpoint_path)
    summary: dict[str, Any] | None = None
    if summary_path.is_file():
        terminal = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = {
            "path": _relative(summary_path, root),
            "sha256": sha256_file(summary_path),
            "terminal_epoch": terminal.get("epoch"),
            "best_epoch": terminal.get("best_epoch"),
            "best_metric_value": terminal.get("best_metric_value"),
            "selection_metric": terminal.get("selection_metric"),
            "selection_mode": terminal.get("selection_mode"),
        }

    provenance = payload.get("provenance") if isinstance(payload.get("provenance"), dict) else {}
    release = provenance.get("release") if isinstance(provenance.get("release"), dict) else None
    release_selection = release.get("selection") if release and isinstance(release.get("selection"), dict) else None
    release_dataset = release.get("dataset") if release and isinstance(release.get("dataset"), dict) else None
    record = {
        "path": _relative(checkpoint_path, root),
        "role": "last" if checkpoint_path.name.endswith(".last.pt") else "selected",
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_schema_version": payload.get("schema_version"),
        "model_name": payload.get("model_name"),
        "model_config": payload.get("model_config", {}),
        "parameter_count": parameter_count,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "task": metrics.get("task") or payload.get("model_config", {}).get("task"),
        "selection": release_selection or {
            "metric": metrics.get("selection_metric"),
            "mode": metrics.get("selection_mode"),
            "value": metrics.get("selection_value"),
        },
        "training_metrics": _scalar_metrics(training),
        "validation_metrics": _scalar_metrics(validation),
        "training_dataset": (
            {
                "manifest_path": dataset["manifest_path"],
                "manifest_sha256": dataset["manifest_sha256"],
                "content_sha256": dataset["content_sha256"],
            }
            if dataset
            else {
                "manifest_path": None,
                "manifest_sha256": None,
                "content_sha256": _dataset_content_sha256(embedded_chunks) if embedded_chunks else None,
            }
        ),
        "summary": summary,
        "release_provenance": release,
    }
    if release_dataset:
        record["training_dataset"] = release_dataset
    return record


def build_registry(
    root: Path,
    data_dir: Path,
    models_dir: Path,
    pattern: str,
    release_manifest_path: Path | None = None,
) -> dict[str, Any]:
    datasets: list[dict[str, Any]] = []
    dataset_by_fingerprint: dict[tuple[str, ...], dict[str, Any]] = {}
    for manifest_path in sorted(data_dir.glob("mesh_v2*/manifest.json")):
        record, fingerprint = _dataset_record(manifest_path, root)
        datasets.append(record)
        dataset_by_fingerprint[fingerprint] = record

    release_manifest = None
    if release_manifest_path is not None and release_manifest_path.is_file():
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        checkpoint_paths = [root / entry["path"] for entry in release_manifest.get("checkpoints", [])]
    else:
        checkpoint_paths = sorted(models_dir.glob(pattern))
    checkpoints = [_checkpoint_record(path, root, dataset_by_fingerprint) for path in checkpoint_paths if path.is_file()]
    source_snapshots = {
        json.dumps(checkpoint["release_provenance"]["source_snapshot"], sort_keys=True): checkpoint["release_provenance"]["source_snapshot"]
        for checkpoint in checkpoints
        if checkpoint.get("release_provenance") and checkpoint["release_provenance"].get("source_snapshot")
    }
    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hash_algorithm": "SHA-256",
        "content_sha256_definition": "SHA-256 of canonical JSON containing ordered chunk path, sample count, and declared SHA-256",
        "datasets": datasets,
        "checkpoints": checkpoints,
        "release_manifest": (
            {
                "path": _relative(release_manifest_path, root),
                "sha256": sha256_file(release_manifest_path),
            }
            if release_manifest_path is not None and release_manifest_path.is_file()
            else None
        ),
        "source_snapshots": list(source_snapshots.values()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--pattern", default="*mesh_v2*.pt")
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_dir = (args.data_dir or project_root / "data").resolve()
    models_dir = (args.models_dir or project_root / "models").resolve()
    output_path = args.out or project_root / "artifacts" / "software_completion_20260713" / "checkpoint_registry.json"
    release_manifest = args.release_manifest
    if release_manifest is None:
        candidate = project_root / "configs" / "release_checkpoints.json"
        release_manifest = candidate if candidate.is_file() else None
    registry = build_registry(project_root, data_dir, models_dir, args.pattern, release_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "datasets": len(registry["datasets"]),
                "checkpoints": len(registry["checkpoints"]),
            }
        )
    )


if __name__ == "__main__":
    main()
