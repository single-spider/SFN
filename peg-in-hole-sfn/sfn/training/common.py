from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_checkpoint_compatible(
    checkpoint: dict,
    expected: dict[str, Any],
    *,
    allow_incompatible: bool = False,
) -> None:
    """Reject missing or mismatched runtime compatibility metadata."""
    if allow_incompatible:
        return
    recorded = checkpoint.get("compatibility") or {}
    problems = []
    for key, expected_value in expected.items():
        if expected_value is None:
            continue
        if key not in recorded or recorded[key] is None:
            problems.append(f"{key}=missing (expected {expected_value!r})")
        elif recorded[key] != expected_value:
            problems.append(f"{key}={recorded[key]!r} (expected {expected_value!r})")
    if problems:
        raise ValueError(
            "Checkpoint/runtime compatibility failure: "
            + "; ".join(problems)
            + ". Use the explicit diagnostic override only when investigating an incompatible combination."
        )


def git_status(repo: str | Path | None = None) -> dict[str, Any]:
    cwd = Path(repo or ".")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def run_metadata(seed: int, argv: list[str] | None = None) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "argv": list(sys.argv if argv is None else argv),
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp_unix": time.time(),
        "git": git_status(),
    }


def make_checkpoint(
    model_name: str,
    model_config: dict,
    model_state_dict: dict,
    optimizer_state_dict=None,
    scheduler_state_dict=None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: dict | None = None,
    data_split: dict | None = None,
    run: dict | None = None,
    train_config: dict | None = None,
    compatibility: dict | None = None,
    training_state: dict | None = None,
) -> dict:
    # Keep schema v1 and all historical keys so old readers/checkpoints remain
    # compatible; metadata fields are additive.
    return {
        "schema_version": 1,
        "model_name": model_name,
        "model_config": dict(model_config),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": metrics or {},
        "data_split": data_split or {},
        "run": run or {},
        "train_config": train_config or {},
        "compatibility": compatibility or {},
        "training_state": training_state or {},
    }


def save_checkpoint(path: str | Path, checkpoint: dict) -> None:
    import numpy as np
    import torch

    def safe_value(value):
        """Keep new checkpoints compatible with PyTorch's safe loader.

        Deterministic resume state naturally contains NumPy arrays/scalars.
        Encoding those as tensors/primitives avoids pickled NumPy globals while
        preserving the exact values needed to resume a local training run.
        """
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value.copy())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: safe_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(safe_value(item) for item in value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(safe_value(checkpoint), path)


def load_checkpoint_cpu(path: str | Path) -> dict:
    import torch

    # Training checkpoints intentionally contain optimizer and deterministic
    # resume state (including NumPy RNG/environment arrays), not only tensors.
    # PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default,
    # which rejects that trusted, locally-produced state.  Checkpoint callers
    # already opt into executing a project checkpoint path, so make the legacy
    # full-checkpoint contract explicit here rather than failing implicitly.
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing model_state_dict")
    return ckpt
