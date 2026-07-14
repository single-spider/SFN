#!/usr/bin/env python
"""Summarize one or two evaluation ``episodes.csv`` artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.evaluation.reporting import read_episode_csv, summarize_benchmarks


def _input(value: str) -> tuple[str | None, Path]:
    """Parse either PATH or LABEL=PATH without breaking Windows drive letters."""
    if "=" in value:
        label, path = value.split("=", 1)
        if not label:
            raise argparse.ArgumentTypeError("input label may not be empty")
        return label, Path(path)
    return None, Path(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=_input, metavar="[LABEL=]EPISODES.csv")
    parser.add_argument("--output", "-o", type=Path, help="Write JSON here (stdout is always populated).")
    parser.add_argument("--pair-keys", default="shape,episode,task", help="Comma-separated episode identity columns.")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    rows: list[dict] = []
    for label, path in args.inputs:
        rows.extend(read_episode_csv(path, method=label))
    keys = tuple(key.strip() for key in args.pair_keys.split(",") if key.strip())
    report = summarize_benchmarks(rows, pair_keys=keys, resamples=args.resamples, seed=args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
