from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from sfn.panda.validation import validate_command_tracking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="square-concave1")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="artifacts/panda_validation/command_tracking_smoke")
    args = ap.parse_args()
    _, summary = validate_command_tracking(args.shape, args.trials, args.out, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
