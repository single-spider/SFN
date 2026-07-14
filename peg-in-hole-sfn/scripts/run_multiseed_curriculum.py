#!/usr/bin/env python
"""Run independent SFMS curricula and aggregate their selected validation metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _seeds(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(set(values)) != len(values) or not values:
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique comma-separated list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_seeds, default=_seeds("101,102,103"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/sfms_multiseed"))
    parser.add_argument("--curriculum", type=Path, default=Path("configs/sfms_curriculum.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/mesh_insertion_tight.yaml"))
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--orientation", type=Path, required=True)
    parser.add_argument("--initial-policy", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        destination = args.out / f"seed_{seed}"
        command = [
            sys.executable,
            "scripts/train_sfms_curriculum.py",
            "--curriculum",
            str(args.curriculum),
            "--config",
            str(args.config),
            "--out",
            str(destination),
            "--seed",
            str(seed),
            "--segmentation",
            str(args.segmentation),
            "--position",
            str(args.position),
            "--orientation",
            str(args.orientation),
        ]
        if args.initial_policy is not None:
            command.extend(["--initial-policy", str(args.initial_policy)])
        subprocess.run(command, check=True)
        summary = json.loads((destination / "curriculum_summary.json").read_text(encoding="utf-8"))
        final = summary["stages"][-1]
        evaluation = final["metrics"].get("best_eval") or final["metrics"].get("eval") or {}
        rows.append(
            {
                "seed": seed,
                "checkpoint": summary["selected_checkpoint"],
                "success_rate": evaluation.get("success_rate"),
                "mean_final_xy_error_mm": evaluation.get("mean_final_xy_error_mm"),
                "mean_final_yaw_error_deg": evaluation.get("mean_final_yaw_error_deg"),
            }
        )
    aggregate = {"seeds": args.seeds, "runs": rows}
    for field in ("success_rate", "mean_final_xy_error_mm", "mean_final_yaw_error_deg"):
        values = [float(row[field]) for row in rows if row[field] is not None]
        if values:
            aggregate[field] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    (args.out / "multiseed_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
