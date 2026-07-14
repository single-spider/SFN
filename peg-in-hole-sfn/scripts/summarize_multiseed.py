#!/usr/bin/env python
"""Aggregate repeated evaluation CSV files across independent random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    return None if not value else float(value)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total == 0:
        return {"low": 0.0, "high": 0.0}
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {"low": max(0.0, centre - radius), "high": min(1.0, centre + radius)}


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.fmean(values))


def summarize(root: Path) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    files = sorted(root.rglob("episodes.csv"))
    for path in files:
        seed_name = next((part for part in path.parts if part.startswith("seed_")), "unknown")
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                method = row.get("method", "").strip() or path.parent.name
                grouped[method].append((seed_name, row))

    methods: dict[str, Any] = {}
    for method, tagged_rows in sorted(grouped.items()):
        by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
        for seed, row in tagged_rows:
            by_seed[seed].append(row)
        rows = [row for _, row in tagged_rows]
        successes = sum(row.get("success", "").lower() in {"1", "true", "yes"} for row in rows)
        xy = [value for row in rows if (value := _as_float(row, "final_xy_error_mm")) is not None]
        yaw = [value for row in rows if (value := _as_float(row, "final_yaw_error_deg")) is not None]
        steps = [value for row in rows if (value := _as_float(row, "steps")) is not None]
        per_seed: dict[str, Any] = {}
        for seed, seed_rows in sorted(by_seed.items()):
            seed_successes = sum(
                row.get("success", "").lower() in {"1", "true", "yes"} for row in seed_rows
            )
            seed_xy = [
                value
                for row in seed_rows
                if (value := _as_float(row, "final_xy_error_mm")) is not None
            ]
            seed_yaw = [
                value
                for row in seed_rows
                if (value := _as_float(row, "final_yaw_error_deg")) is not None
            ]
            per_seed[seed] = {
                "episodes": len(seed_rows),
                "successes": seed_successes,
                "success_rate": seed_successes / len(seed_rows),
                "mean_final_xy_error_mm": _mean(seed_xy),
                "mean_final_yaw_error_deg": _mean(seed_yaw),
            }
        methods[method] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "success_rate_wilson_95": _wilson(successes, len(rows)),
            "mean_final_xy_error_mm": _mean(xy),
            "mean_final_yaw_error_deg": _mean(yaw),
            "mean_steps": _mean(steps),
            "seed_count": len(by_seed),
            "per_seed": per_seed,
        }
    return {"schema_version": 1, "root": str(root.resolve()), "csv_files": len(files), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
