"""Subprocess tests for the supported command-line help contract."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPTS = sorted(
    path.relative_to(ROOT)
    for path in (ROOT / "scripts").glob("*.py")
    if path.name != "smoke_software.py" and "ArgumentParser" in path.read_text(encoding="utf-8")
)


@pytest.mark.parametrize("script", CLI_SCRIPTS, ids=lambda path: path.stem)
def test_cli_help_exits_successfully(script: Path) -> None:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"})
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_smoke_driver_help_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_software.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--full" in result.stdout
