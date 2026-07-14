#!/usr/bin/env python
"""Run the staged SFMS curriculum defined by a YAML file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sfn.config import load_config
from sfn.constants import DEFAULT_SHAPE_SPLITS
from sfn.training.curriculum import run_sfms_curriculum, stages_from_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, default=Path("configs/sfms_curriculum.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/mesh_insertion_tight.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/sfms_curriculum"))
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-split", choices=sorted(DEFAULT_SHAPE_SPLITS), default="train_seen")
    parser.add_argument("--eval-split", choices=sorted(DEFAULT_SHAPE_SPLITS), default="validation_unseen")
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--orientation", type=Path, required=True)
    parser.add_argument("--initial-policy", type=Path, default=None)
    args = parser.parse_args()
    raw = yaml.safe_load(args.curriculum.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("curriculum root must be a mapping")
    stages = stages_from_mapping(raw)
    config = load_config(args.config, seed=args.seed)
    report = run_sfms_curriculum(
        out_dir=args.out,
        stages=stages,
        seed=args.seed,
        shapes=list(DEFAULT_SHAPE_SPLITS[args.train_split]),
        eval_shapes=list(DEFAULT_SHAPE_SPLITS[args.eval_split]),
        environment=config.environment,
        camera=config.camera,
        segmentation_path=args.segmentation,
        position_path=args.position,
        orientation_path=args.orientation,
        initial_policy_path=args.initial_policy,
        device=config.project.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
