#!/usr/bin/env python
"""Generate all-shape standalone insertion tolerance and convergence maps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sfn.config import InsertionConfig
from sfn.constants import ALL_EXPECTED_SHAPES
from sfn.envs.pybullet_insertion import simulate_pybullet_insertion


def _values(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="all")
    parser.add_argument("--xy-mm", type=_values, default=_values("-1,-0.5,0,0.5,1"))
    parser.add_argument("--yaw-deg", type=_values, default=_values("-1,0,1"))
    parser.add_argument("--target-depth-mm", type=float, default=3.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    shapes = list(ALL_EXPECTED_SHAPES) if args.shapes == "all" else [x.strip() for x in args.shapes.split(",")]
    config = InsertionConfig(target_depth_mm=args.target_depth_mm, descent_increment_mm=0.2, max_descent_attempts=40)
    rows = []
    for shape in shapes:
        for yaw in args.yaw_deg:
            for dy in args.xy_mm:
                for dx in args.xy_mm:
                    result = simulate_pybullet_insertion(shape, [dx / 1000.0, dy / 1000.0, yaw], config)
                    rows.append({"shape": shape, "dx_mm": dx, "dy_mm": dy, "yaw_deg": yaw, **result.to_dict()})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_shape = {}
    for shape in shapes:
        selected = [row for row in rows if row["shape"] == shape]
        per_shape[shape] = {
            "trials": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
            "success_rate": sum(bool(row["success"]) for row in selected) / len(selected),
        }
    summary = {
        "backend": "standalone_pybullet_raster_compound",
        "shapes": len(shapes),
        "trials": len(rows),
        "xy_mm": args.xy_mm,
        "yaw_deg": args.yaw_deg,
        "target_depth_mm": args.target_depth_mm,
        "per_shape": per_shape,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    zero_yaw = [row for row in rows if row["yaw_deg"] == min(args.yaw_deg, key=abs)]
    grid = np.zeros((len(args.xy_mm), len(args.xy_mm)), dtype=float)
    for row in zero_yaw:
        y = args.xy_mm.index(row["dy_mm"])
        x = args.xy_mm.index(row["dx_mm"])
        grid[y, x] += float(row["success"])
    grid /= len(shapes)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(grid, origin="lower", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(args.xy_mm)), labels=args.xy_mm)
    axis.set_yticks(range(len(args.xy_mm)), labels=args.xy_mm)
    axis.set_xlabel("Initial dx (mm)")
    axis.set_ylabel("Initial dy (mm)")
    axis.set_title("Physical insertion basin at nearest-to-zero yaw")
    figure.colorbar(image, ax=axis, label="Success fraction across shapes")
    figure.tight_layout()
    figure.savefig(args.out / "basin_xy.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"trials": len(rows), "shapes": len(shapes)}, indent=2))


if __name__ == "__main__":
    main()
