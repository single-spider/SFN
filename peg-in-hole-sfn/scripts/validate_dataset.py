#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sfn.data import validate_dataset


def main():
    ap = argparse.ArgumentParser(description="Validate a collected SFN dataset directory")
    ap.add_argument("dataset")
    args = ap.parse_args()
    manifest = validate_dataset(args.dataset)
    print(json.dumps({"ok": True, "samples": manifest.get("samples"), "split": manifest.get("split")}, indent=2))


if __name__ == "__main__":
    main()
