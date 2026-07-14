from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from sfn.panda.validation import validate_attachment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="square-concave1")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default="artifacts/panda_validation/attachment_smoke")
    args = ap.parse_args()
    _, summary = validate_attachment(args.shape, args.steps, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
