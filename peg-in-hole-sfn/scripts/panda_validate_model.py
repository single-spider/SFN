from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from sfn.panda.validation import validate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="square-concave1")
    ap.add_argument("--out", default="artifacts/panda_validation/model_validation_smoke")
    args = ap.parse_args()
    _, summary = validate_model([s for s in args.shapes.split(",") if s], args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
