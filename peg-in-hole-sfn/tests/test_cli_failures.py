"""Subprocess checks that supported CLIs fail closed when inputs are missing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"})
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _assert_failed_for(result: subprocess.CompletedProcess[str], input_name: str) -> None:
    assert result.returncode != 0, result.stdout
    assert input_name.lower() in (result.stdout + result.stderr).lower()


def test_data_validation_cli_fails_for_missing_dataset(tmp_path: Path) -> None:
    missing = tmp_path / "missing-data-input"
    _assert_failed_for(_run("scripts/validate_dataset.py", str(missing)), missing.name)


@pytest.mark.parametrize(
    "script",
    ["train_segmentation.py", "train_position.py", "train_orientation.py"],
)
def test_training_cli_fails_for_missing_dataset(script: str, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    missing = tmp_path / f"missing-{Path(script).stem}-input"
    result = _run(
        f"scripts/{script}",
        "--dataset",
        str(missing),
        "--out",
        str(tmp_path / "unused.pt"),
        "--epochs",
        "1",
        "--no-progress",
    )
    _assert_failed_for(result, missing.name)


def test_perception_evaluation_cli_fails_for_missing_dataset(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    missing_dataset = tmp_path / "missing-evaluation-dataset"
    missing_checkpoint = tmp_path / "missing-evaluation-checkpoint.pt"
    result = _run(
        "scripts/evaluate_perception.py",
        "--dataset",
        str(missing_dataset),
        "--segmentation",
        str(missing_checkpoint),
        "--out",
        str(tmp_path / "unused.json"),
    )
    _assert_failed_for(result, missing_dataset.name)


def test_controller_evaluation_cli_requires_policy_input() -> None:
    result = _run("scripts/evaluate.py", "--method", "sfms")
    assert result.returncode != 0
    assert "--sfms-policy" in result.stderr
