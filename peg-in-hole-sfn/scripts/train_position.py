#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.training.perception_cli import add_perception_args, print_result, run_perception_cli


def main():
    ap = argparse.ArgumentParser(description="Train position heatmap model on collected SFN NPZ data")
    add_perception_args(ap, root=ROOT, task="position", default_config="position.yaml")
    args = ap.parse_args()
    print_result(run_perception_cli("position", args))


if __name__ == "__main__":
    main()
