#!/usr/bin/env python
"""Atomically add release provenance to selected checkpoints without changing weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfn.training.common import file_sha256, load_checkpoint_cpu, save_checkpoint

DEFAULT_RELEASE_MANIFEST = ROOT / "configs" / "release_checkpoints.json"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Hash tensor names, types, shapes, and raw values independent of torch serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8") + b"\0")
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(canonical_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _scalar_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _selection(checkpoint: dict[str, Any], declared: dict[str, Any] | None) -> dict[str, Any]:
    declared = dict(declared or {})
    metrics = checkpoint.get("metrics") if isinstance(checkpoint.get("metrics"), dict) else {}
    validation = metrics.get("val") if isinstance(metrics.get("val"), dict) else {}
    metric = declared.get("metric", metrics.get("selection_metric"))
    value = metrics.get("selection_value")
    if metric and value is None:
        value = _nested(metrics, metric)
    result = {
        "status": declared.get("status", "known" if metric is not None and value is not None else "unknown"),
        "metric": metric,
        "mode": declared.get("mode", metrics.get("selection_mode")),
        "value": value,
        "selected_epoch": checkpoint.get("epoch"),
        "validation_metrics": _scalar_metrics(validation),
        "source": "checkpoint_metrics" if metric is not None else "release_manifest",
    }
    if declared.get("reason"):
        result["reason"] = declared["reason"]
    return result


def _dataset(root: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    split = checkpoint.get("data_split") if isinstance(checkpoint.get("data_split"), dict) else {}
    chunks = split.get("chunks") if isinstance(split.get("chunks"), list) else []
    fingerprint = tuple(str(chunk.get("sha256")) for chunk in chunks if isinstance(chunk, dict))
    content_sha256 = canonical_sha256(
        [
            {"path": str(chunk.get("path", "")).replace("\\", "/"), "samples": chunk.get("samples"), "sha256": chunk.get("sha256")}
            for chunk in chunks
            if isinstance(chunk, dict)
        ]
    ) if chunks else None
    for manifest_path in sorted((root / "data").glob("**/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_chunks = manifest.get("chunks") if isinstance(manifest.get("chunks"), list) else []
        candidate = tuple(str(chunk.get("sha256")) for chunk in manifest_chunks if isinstance(chunk, dict))
        if fingerprint and candidate == fingerprint:
            return {
                "status": "known",
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": file_sha256(manifest_path),
                "content_sha256": content_sha256,
            }
    return {
        "status": "embedded_only" if chunks else "unknown",
        "manifest_path": None,
        "manifest_sha256": None,
        "content_sha256": content_sha256,
    }


def build_release_provenance(
    root: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    entry: dict[str, Any],
    source_snapshot_path: Path,
) -> dict[str, Any]:
    snapshot = json.loads(source_snapshot_path.read_text(encoding="utf-8"))
    config_files = []
    for raw_path in entry.get("config_paths", []):
        config_path = root / raw_path
        config_files.append({"path": Path(raw_path).as_posix(), "sha256": file_sha256(config_path)})
    model_config = checkpoint.get("model_config", {})
    train_config = checkpoint.get("train_config", {})
    previous = checkpoint.get("provenance") if isinstance(checkpoint.get("provenance"), dict) else {}
    previous_release = previous.get("release") if isinstance(previous.get("release"), dict) else {}
    stamped_from = previous_release.get("stamped_from") or {
        "path": checkpoint_path.relative_to(root).as_posix() if checkpoint_path.is_relative_to(root) else checkpoint_path.as_posix(),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    return {
        "schema_version": 1,
        "component": entry["component"],
        "source_snapshot": {
            "manifest_path": source_snapshot_path.relative_to(root).as_posix() if source_snapshot_path.is_relative_to(root) else source_snapshot_path.as_posix(),
            "manifest_sha256": file_sha256(source_snapshot_path),
            "aggregate_sha256": snapshot["aggregate_sha256"],
            "file_count": snapshot["file_count"],
            "git_head": snapshot.get("git_head"),
        },
        "configuration": {
            "model_config_sha256": canonical_sha256(model_config),
            "train_config_sha256": canonical_sha256(train_config),
            "release_config_files": config_files,
        },
        "dataset": _dataset(root, checkpoint),
        "selection": _selection(checkpoint, entry.get("selection")),
        "weights_sha256": state_dict_sha256(checkpoint.get("model_state_dict", {})),
        "stamped_from": stamped_from,
    }


def stamp_checkpoint(path: Path, checkpoint: dict[str, Any], release: dict[str, Any]) -> str:
    """Write and verify an additive metadata migration, then atomically replace path."""
    before = state_dict_sha256(checkpoint.get("model_state_dict", {}))
    existing = checkpoint.get("provenance") if isinstance(checkpoint.get("provenance"), dict) else {}
    if existing.get("release") == release:
        return "unchanged"
    checkpoint["provenance"] = {**existing, "release": release}
    temporary = path.with_name(f".{path.name}.provenance.tmp")
    try:
        save_checkpoint(temporary, checkpoint)
        verified = load_checkpoint_cpu(temporary)
        after = state_dict_sha256(verified.get("model_state_dict", {}))
        if after != before:
            raise RuntimeError(f"Weight verification failed for {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return "stamped"


def stamp_release(
    root: Path,
    release_manifest_path: Path,
    source_snapshot_path: Path,
    *,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("checkpoints", [])
    dependencies: dict[str, str] = {}
    results = []
    for entry in entries:
        source = root / entry["path"]
        destination = source if output_dir is None else output_dir / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if output_dir is not None and destination != source:
            destination.write_bytes(source.read_bytes())
        checkpoint = load_checkpoint_cpu(destination)
        if entry["component"] in {"sfms", "mfms"}:
            compatibility = checkpoint.get("compatibility") if isinstance(checkpoint.get("compatibility"), dict) else {}
            checkpoint["compatibility"] = {
                **compatibility,
                "segmentation_sha256": dependencies.get("segmentation"),
                "position_sha256": dependencies.get("position"),
                "orientation_sha256": dependencies.get("orientation"),
            }
        release = build_release_provenance(root, destination, checkpoint, entry, source_snapshot_path)
        action = stamp_checkpoint(destination, checkpoint, release)
        dependencies[entry["component"]] = file_sha256(destination) or ""
        results.append(
            {
                "component": entry["component"],
                "path": destination.relative_to(root).as_posix() if destination.is_relative_to(root) else destination.as_posix(),
                "action": action,
                "checkpoint_sha256": file_sha256(destination),
                "weights_sha256": release["weights_sha256"],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--release-manifest", type=Path, default=DEFAULT_RELEASE_MANIFEST)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Stamp copies instead of the selected files in place")
    parser.add_argument("--in-place", action="store_true", help="Atomically update the selected checkpoint files")
    args = parser.parse_args()
    if args.in_place == (args.output_dir is not None):
        parser.error("choose exactly one of --in-place or --output-dir")
    root = args.project_root.resolve()
    results = stamp_release(root, args.release_manifest.resolve(), args.source_snapshot.resolve(), output_dir=args.output_dir)
    print(json.dumps({"checkpoints": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
