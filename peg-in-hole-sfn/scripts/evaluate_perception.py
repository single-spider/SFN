#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.evaluation.evaluate_perception import evaluate_all, write_metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate SFN perception checkpoints on a collected dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--position", default=None)
    ap.add_argument("--orientation", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--per-shape", action="store_true", help="Include per-shape metrics for shape-disjoint analysis.")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "perception_metrics.json"))
    args = ap.parse_args()
    metrics = evaluate_all(
        args.dataset, args.segmentation, args.position, args.orientation, args.limit, per_shape=args.per_shape
    )
    write_metrics(metrics, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
