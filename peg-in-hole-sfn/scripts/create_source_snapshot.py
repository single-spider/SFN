#!/usr/bin/env python
"""Create a deterministic source manifest and portable ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INCLUDED_ROOTS = (".github", "algos", "configs", "docs", "gymEnv", "scripts", "sfn", "tests", "utils")
INCLUDED_FILES = (
    "collect_sim_samples.py",
    "demo_closed_loop_gui.py",
    "demo_object_insertion_gui.py",
    "experiment_1.py",
    "HANDOVER.md",
    "pyproject.toml",
    "pytest.ini",
    "README.md",
    "simple_closed_loop_sim.py",
    "test.py",
    "testEnv.py",
    "test_pose_gui_8.py",
    "test_pose_position_gui_11.py",
    "TRAINING_PIPELINE.md",
    "train_pose_8.py",
    "train_position_11.py",
    "view_pybullet_scene.py",
    "vis_pose_8.py",
)
EXCLUDED_PARTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
MANIFEST_NAME = "SOURCE_SNAPSHOT.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(root: Path, command: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *command], cwd=root, text=True, capture_output=True, check=False
    )
    value = result.stdout.strip()
    return value or None


def _excluded(path: Path) -> bool:
    return bool(EXCLUDED_PARTS.intersection(path.parts)) or path.name == ".DS_Store" or path.name.startswith("._")


def source_files(root: Path) -> list[Path]:
    """Return the stable, release-relevant file set in archive path order."""
    files: set[Path] = set()
    for directory in INCLUDED_ROOTS:
        candidate = root / directory
        if candidate.is_dir():
            files.update(path for path in candidate.rglob("*") if path.is_file() and not _excluded(path.relative_to(root)))
    files.update(root / name for name in INCLUDED_FILES if (root / name).is_file())
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_snapshot(root: Path, files: Iterable[Path] | None = None) -> dict[str, Any]:
    selected = list(source_files(root) if files is None else files)
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in selected
    ]
    canonical_records = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 2,
        "hash_algorithm": "SHA-256",
        "aggregate_definition": "SHA-256 of canonical JSON for the ordered files array",
        "aggregate_sha256": _sha256_bytes(canonical_records),
        "git_head": _git(root, ["rev-parse", "HEAD"]),
        "file_count": len(records),
        "files": records,
    }


def snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_archive(root: Path, archive_path: Path, files: Iterable[Path], manifest: bytes) -> None:
    """Write a reproducible ZIP (fixed timestamps, order, permissions, and paths)."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    epoch = max(315532800, int(os.environ.get("SOURCE_DATE_EPOCH", "315532800")))
    timestamp = tuple(__import__("time").gmtime(epoch)[:6])
    entries = [(path.relative_to(root).as_posix(), path.read_bytes(), path) for path in files]
    entries.append((MANIFEST_NAME, manifest, None))
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload, source in sorted(entries, key=lambda item: item[0]):
            info = zipfile.ZipInfo(name, timestamp)
            info.create_system = 3
            executable = source is not None and bool(source.stat().st_mode & stat.S_IXUSR)
            info.external_attr = ((0o755 if executable else 0o644) & 0xFFFF) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(archive_path)


def create_outputs(root: Path, manifest_path: Path, archive_path: Path | None) -> dict[str, Any]:
    files = source_files(root)
    snapshot = build_snapshot(root, files)
    payload = snapshot_bytes(snapshot)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload)
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_bytes(payload),
        "file_count": snapshot["file_count"],
        "aggregate_sha256": snapshot["aggregate_sha256"],
        "git_head": snapshot["git_head"],
    }
    if archive_path is not None:
        write_archive(root, archive_path, files, payload)
        result.update(
            archive=str(archive_path),
            archive_bytes=archive_path.stat().st_size,
            archive_sha256=digest(archive_path),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True, help="Snapshot JSON path")
    parser.add_argument("--archive", type=Path, help="Optional deterministic portable ZIP path")
    args = parser.parse_args()
    print(json.dumps(create_outputs(args.project_root.resolve(), args.out, args.archive), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
