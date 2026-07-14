#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sfn.config import InsertionConfig
from sfn.constants import ALL_EXPECTED_SHAPES
from sfn.envs.pybullet_insertion import simulate_pybullet_insertion


def main():
    ap = argparse.ArgumentParser(description="Validate true standalone PyBullet mesh insertion.")
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--dx-mm", type=float, default=0.0)
    ap.add_argument("--dy-mm", type=float, default=0.0)
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--target-depth-mm", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shapes = ALL_EXPECTED_SHAPES if a.shapes == "all" else [x.strip() for x in a.shapes.split(",")]
    cfg = InsertionConfig(target_depth_mm=a.target_depth_mm, descent_increment_mm=0.2, max_descent_attempts=40)
    rows = [
        {"shape": s, **simulate_pybullet_insertion(s, [a.dx_mm / 1000, a.dy_mm / 1000, a.yaw_deg], cfg).to_dict()}
        for s in shapes
    ]
    report = {
        "trials": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "pose": [a.dx_mm, a.dy_mm, a.yaw_deg],
        "rows": rows,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
