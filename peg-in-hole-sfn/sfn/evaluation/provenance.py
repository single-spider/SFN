"""Portable run manifests and immutable input identifiers."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ..training.common import run_metadata


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    result = {}
    for package in ("numpy", "torch", "gymnasium", "pybullet", "opencv-python", "Pillow"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def write_run_provenance(
    out_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    arguments: dict[str, Any],
    input_paths: dict[str, str | Path | None],
    seed: int,
    backend: str,
) -> dict[str, Any]:
    """Write resolved configuration and a manifest with hashes of every input."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "resolved_config.json"
    config_path.write_text(json.dumps(resolved_config, indent=2, default=str) + "\n", encoding="utf-8")
    inputs = {}
    for name, raw_path in input_paths.items():
        if raw_path in (None, ""):
            inputs[name] = None
            continue
        path = Path(raw_path)
        inputs[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    manifest = {
        "schema_version": 1,
        "backend": backend,
        "arguments": arguments,
        "resolved_config": config_path.name,
        "resolved_config_sha256": sha256_file(config_path),
        "inputs": inputs,
        "runtime": {**run_metadata(seed), "packages": _package_versions()},
    }
    (destination / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest
