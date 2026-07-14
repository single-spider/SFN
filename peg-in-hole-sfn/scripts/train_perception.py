#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.training.perception_cli import add_perception_args, print_result, run_perception_cli


def main():
    # Parse task first with no help so `--help` can show the full option set.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", default="segmentation", choices=["segmentation", "position", "orientation"])
    task_args, _ = pre.parse_known_args()
    default_cfg = {
        "segmentation": "segmentation.yaml",
        "position": "position.yaml",
        "orientation": "orientation.yaml",
    }[task_args.task]
    ap = argparse.ArgumentParser(description="Train any SFN perception model", parents=[pre])
    add_perception_args(ap, root=ROOT, task=task_args.task, default_config=default_cfg)
    args = ap.parse_args()
    print_result(run_perception_cli(args.task, args))


if __name__ == "__main__":
    main()
