from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from sfn.data.splits import get_split
from sfn.panda.validation import evaluate_oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--shapes", default=None)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="artifacts/panda_validation/oracle_alignment_smoke")
    args = ap.parse_args()
    shapes = [s for s in args.shapes.split(",") if s] if args.shapes else get_split(args.split or "test_unseen")
    _, summary = evaluate_oracle(shapes, args.episodes, args.seed, out_dir=args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
