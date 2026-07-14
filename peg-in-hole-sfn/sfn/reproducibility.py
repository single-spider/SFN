"""Run, dataset, checkpoint, and source-state reproducibility helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: str | Path) -> dict[str, Any]:
    root = Path(root)

    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        status = run("status", "--porcelain")
        return {
            "root": run("rev-parse", "--show-toplevel"),
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"root": None, "commit": None, "branch": None, "dirty": None, "status_porcelain": []}


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "gymnasium",
                "matplotlib",
                "numpy",
                "opencv-python",
                "pillow",
                "pybullet",
                "PyYAML",
                "pytest",
                "scipy",
                "torch",
                "torchvision",
                "tqdm",
                "trimesh",
            ]
        ),
    }
    try:
        import torch

        state["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # diagnostic path must survive a broken torch install
        state["torch"] = {"error": str(exc)}
    return state


def dataset_inventory(data_root: str | Path) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    rows = []
    if not data_root.exists():
        return rows
    for manifest_path in sorted(data_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "path": str(manifest_path.parent),
                    "manifest_sha256": sha256_file(manifest_path),
                    "schema_version": manifest.get("schema_version"),
                    "dataset_type": manifest.get("dataset_type", "cartesian_legacy"),
                    "renderer_backend": manifest.get("renderer_backend", "unknown_legacy"),
                    "split": manifest.get("split"),
                    "samples": manifest.get("samples"),
                    "shapes": manifest.get("shapes"),
                    "chunks": len(manifest.get("chunks") or []),
                }
            )
        except Exception as exc:
            rows.append({"path": str(manifest_path.parent), "error": str(exc)})
    return rows


def checkpoint_inventory(model_root: str | Path) -> list[dict[str, Any]]:
    model_root = Path(model_root)
    rows = []
    if not model_root.exists():
        return rows
    import torch

    for path in sorted(model_root.glob("*.pt")):
        row: dict[str, Any] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                row.update(
                    {
                        "schema_version": checkpoint.get("schema_version"),
                        "model_name": checkpoint.get("model_name"),
                        "model_config": checkpoint.get("model_config"),
                        "epoch": checkpoint.get("epoch"),
                        "global_step": checkpoint.get("global_step"),
                        "data_split": checkpoint.get("data_split"),
                    }
                )
            else:
                row["format"] = type(checkpoint).__name__
        except Exception as exc:
            row["load_error"] = str(exc)
        rows.append(row)
    return rows
