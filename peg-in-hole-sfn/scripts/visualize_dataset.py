#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.evaluation.visuals import generate_dataset_visuals


def main():
    ap = argparse.ArgumentParser(description="Generate RGB/mask/target/prediction visual panels")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "visuals"))
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--position", default=None)
    ap.add_argument("--orientation", default=None)
    args = ap.parse_args()
    paths = generate_dataset_visuals(
        args.dataset, args.out, args.count, args.segmentation, args.position, args.orientation
    )
    print(json.dumps({"count": len(paths), "paths": [str(p) for p in paths]}, indent=2))


if __name__ == "__main__":
    main()
