#!/usr/bin/env python
"""Consolidate paired Panda controller runs into one auditable matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.evaluation.statistics import paired_binary_counts, paired_bootstrap_difference, wilson_interval  # noqa: E402


def _boolean(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["success"] = _boolean(row["success"])
        row["steps"] = int(row["steps"])
        row["final_xy_error_mm"] = float(row["final_xy_error_mm"])
        row["final_yaw_error_deg"] = float(row["final_yaw_error_deg"])
    return rows


def _key(row: dict) -> tuple[str, int]:
    return str(row["shape"]), int(row["episode"])


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(row["success"] for row in rows)
    low, high = wilson_interval(successes, n)
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "success_rate_ci95_low": low,
        "success_rate_ci95_high": high,
        "mean_steps": sum(row["steps"] for row in rows) / n,
        "mean_final_xy_error_mm": sum(row["final_xy_error_mm"] for row in rows) / n,
        "mean_final_yaw_error_deg": sum(row["final_yaw_error_deg"] for row in rows) / n,
        "failure_taxonomy": dict(
            sorted(Counter(row.get("termination_reason") or "unknown" for row in rows if not row["success"]).items())
        ),
    }


def _paired_summary(left_rows: list[dict], right_rows: list[dict], *, seed: int) -> dict:
    left = {_key(row): row for row in left_rows}
    right = {_key(row): row for row in right_rows}
    keys = sorted(set(left) & set(right))
    result = paired_binary_counts([left[key]["success"] for key in keys], [right[key]["success"] for key in keys])
    result["continuous_paired_bootstrap"] = {
        field: paired_bootstrap_difference(
            [left[key][field] for key in keys],
            [right[key][field] for key in keys],
            seed=seed + index,
        )
        for index, field in enumerate(("final_xy_error_mm", "final_yaw_error_deg", "steps"))
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts" / "software_completion_20260713")
    parser.add_argument("--prefix", default="panda_matrix_v2_")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root / "panda_dynamic_insertion_matrix"
    out.mkdir(parents=True, exist_ok=True)

    runs: dict[tuple[str, str], list[dict]] = {}
    for directory in sorted(args.root.glob(f"{args.prefix}*")):
        episode_file = directory / "episodes.csv"
        if not episode_file.exists():
            continue
        rows = _read(episode_file)
        if not rows:
            continue
        runs[(str(rows[0]["method"]), str(rows[0]["mask_source"]))] = rows
    expected = {
        (method, mask) for method in ("oracle", "sfss", "sfms", "mfms") for mask in ("ground_truth", "predicted")
    }
    missing = sorted(expected - set(runs))
    if missing:
        raise SystemExit(f"Missing matrix cells: {missing}")

    matrix = []
    for (method, mask), rows in sorted(runs.items(), key=lambda item: (item[0][1], item[0][0])):
        matrix.append({"method": method, "mask_source": mask, **_summary(rows)})

    paired = {}
    for method in ("sfss", "sfms", "mfms"):
        paired[f"{method}:predicted_minus_ground_truth"] = _paired_summary(
            runs[(method, "predicted")], runs[(method, "ground_truth")], seed=7100
        )
    for mask in ("ground_truth", "predicted"):
        for method in ("sfss", "mfms"):
            paired[f"{method}_minus_sfms:{mask}"] = _paired_summary(
                runs[(method, mask)], runs[("sfms", mask)], seed=7200
            )

    report = {
        "backend": "panda_native_camera_dynamic_contact",
        "task": "alignment_then_physical_insertion",
        "test_shapes": sorted({row["shape"] for rows in runs.values() for row in rows}),
        "matrix": matrix,
        "paired_comparisons": paired,
        "observation_note": (
            "Predicted-mask runs use a simulated high-contrast blue peg and a segmentation model trained on "
            "the same randomized PyBullet camera domain; this is not evidence of real-camera transfer."
        ),
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fields = [
        "method",
        "mask_source",
        "episodes",
        "successes",
        "success_rate",
        "success_rate_ci95_low",
        "success_rate_ci95_high",
        "mean_steps",
        "mean_final_xy_error_mm",
        "mean_final_yaw_error_deg",
        "failure_taxonomy",
    ]
    with (out / "matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, "failure_taxonomy": json.dumps(row["failure_taxonomy"])} for row in matrix)

    lines = [
        "# Panda Dynamic Insertion Matrix",
        "",
        "All rows use the two held-out shapes, ten trials per shape, measured Panda motor execution, "
        "and physical downward insertion after the strict alignment threshold is reached.",
        "",
        "| Observation | Method | Success | Wilson 95% CI | Mean XY (mm) | Mean yaw (deg) | Mean steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in matrix:
        lines.append(
            f"| {row['mask_source']} | {row['method'].upper()} | {row['successes']}/{row['episodes']} "
            f"({100 * row['success_rate']:.1f}%) | {100 * row['success_rate_ci95_low']:.1f}–"
            f"{100 * row['success_rate_ci95_high']:.1f}% | {row['mean_final_xy_error_mm']:.3f} | "
            f"{row['mean_final_yaw_error_deg']:.3f} | {row['mean_steps']:.2f} |"
        )
    lines.extend(["", "## Interpretation", "", report["observation_note"], ""])
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "cells": len(matrix), "paired_comparisons": len(paired)}, indent=2))


if __name__ == "__main__":
    main()
