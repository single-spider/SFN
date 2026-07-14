import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from sfn.data.collect import collect_npz

ROOT = Path(__file__).resolve().parents[1]


def _run(args, tmp_path):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )


def test_perception_script_help_surfaces_full_options(tmp_path):
    for script in [
        "scripts/train_segmentation.py",
        "scripts/train_position.py",
        "scripts/train_orientation.py",
        "scripts/train_perception.py",
    ]:
        result = _run([script, "--help"], tmp_path)
        assert "--resume" in result.stdout
        assert "--search-grid" in result.stdout


def test_segmentation_cli_search_smoke(tmp_path):
    data = tmp_path / "data"
    collect_npz(data, split="train_seen", samples_per_shape=1, seed=41)
    out = tmp_path / "search" / "seg.pt"
    result = _run(
        [
            "scripts/train_segmentation.py",
            "--dataset",
            str(data),
            "--out",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--limit",
            "2",
            "--no-progress",
            "--search-grid",
            "lr=0.001",
            "--max-trials",
            "1",
            "--seed",
            "42",
        ],
        tmp_path,
    )
    parsed = json.loads(result.stdout[result.stdout.find("{") :])
    assert parsed["best"]["checkpoint"]
    assert (tmp_path / "search" / "seg" / "search_summary.json").exists()
