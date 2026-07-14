from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from sfn.panda.validation import parse_csv_floats, validate_ik_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="square-concave1")
    ap.add_argument("--grid-mm", default="-10,-5,0,5,10")
    ap.add_argument("--grid-yaw-deg", default="-10,-5,0,5,10")
    ap.add_argument("--out", default="artifacts/panda_validation/ik_grid_smoke")
    args = ap.parse_args()
    _, summary = validate_ik_grid(
        args.shape, parse_csv_floats(args.grid_mm), parse_csv_floats(args.grid_yaw_deg), args.out
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
