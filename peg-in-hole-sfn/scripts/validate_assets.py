#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json

from sfn.config import load_config
from sfn.envs import AssetRegistry


def main():
    ap = argparse.ArgumentParser(description="Validate SFN shape assets")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--shape", action="append")
    ap.add_argument("--strict-dependencies", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, seed=args.seed)
    print(json.dumps(cfg.to_dict(), indent=2))
    results = AssetRegistry().validate_all(args.shape, strict_dependencies=args.strict_dependencies)
    ok = True
    for shape, result in results.items():
        print(f"{shape}: {'OK' if result.valid else 'INVALID'}")
        for w in result.warnings:
            print(f"  warning: {w}")
        for e in result.errors:
            print(f"  error: {e}")
        ok = ok and result.valid
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
