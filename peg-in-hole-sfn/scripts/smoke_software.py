#!/usr/bin/env python3
"""Run the repeatable, CPU-only release smoke checks with one command."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: Sequence[str], *, env: dict[str, str]) -> None:
    printable = subprocess.list2cmdline(list(command))
    print(f"\n==> {printable}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the complete test suite instead of only release and CLI smoke tests.",
    )
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff and mypy checks.")
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    python = sys.executable

    if not args.skip_lint:
        run([python, "-m", "ruff", "check", "sfn", "scripts", "tests"], env=env)
        run([python, "-m", "mypy", "scripts/smoke_software.py"], env=env)

    tests = ["tests"] if args.full else ["tests/test_cli_smoke.py"]
    run([python, "-m", "pytest", "-q", *tests], env=env)
    print("\nCPU software smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
