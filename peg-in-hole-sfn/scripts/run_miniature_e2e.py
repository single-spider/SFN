#!/usr/bin/env python
"""One-command miniature collection-to-Panda release pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/miniature_e2e"))
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train_data, val_data = args.out / "train_data", args.out / "val_data"
    models = args.out / "models"
    models.mkdir(exist_ok=True)
    stages = [
        (
            "collect_train",
            [
                "scripts/collect_dataset.py",
                "--config",
                "configs/mesh_insertion_tight.yaml",
                "--split",
                "train_seen",
                "--samples-per-shape",
                "1",
                "--out",
                train_data,
                "--seed",
                args.seed,
            ],
        ),
        (
            "collect_validation",
            [
                "scripts/collect_dataset.py",
                "--config",
                "configs/mesh_insertion_tight.yaml",
                "--split",
                "validation_unseen",
                "--samples-per-shape",
                "2",
                "--out",
                val_data,
                "--seed",
                args.seed + 1,
            ],
        ),
    ]
    checkpoints = {}
    for task in ("segmentation", "position", "orientation"):
        checkpoint = models / f"{task}.pt"
        checkpoints[task] = checkpoint
        stages.append(
            (
                f"train_{task}",
                [
                    f"scripts/train_{task}.py",
                    "--config",
                    "configs/mesh_insertion_tight.yaml",
                    "--dataset",
                    train_data,
                    "--val-dataset",
                    val_data,
                    "--out",
                    checkpoint,
                    "--epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--base-channels",
                    "4",
                    "--device",
                    "cpu",
                    "--no-progress",
                    "--seed",
                    args.seed + 2,
                ],
            )
        )
    sfms = models / "sfms.pt"
    mfms = models / "mfms.pt"
    common_vsn = [
        "--mask_source",
        "predicted",
        "--segmentation",
        checkpoints["segmentation"],
        "--position",
        checkpoints["position"],
        "--orientation",
        checkpoints["orientation"],
    ]
    stages.extend(
        [
            (
                "train_sfms",
                [
                    "scripts/train_sfms.py",
                    "--config",
                    "configs/mesh_insertion_tight.yaml",
                    "--teacher-pretrain",
                    "--teacher-samples",
                    "8",
                    "--teacher-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--split",
                    "train_seen",
                    "--out",
                    sfms,
                    *common_vsn,
                    "--seed",
                    args.seed + 3,
                ],
            ),
            (
                "train_mfms",
                [
                    "scripts/train_mfms.py",
                    "--config",
                    "configs/mesh_insertion_tight.yaml",
                    "--teacher-pretrain",
                    "--sfms-teacher",
                    sfms,
                    "--teacher-samples",
                    "8",
                    "--teacher-epochs",
                    "1",
                    "--batch-size",
                    "4",
                    "--split",
                    "train_seen",
                    "--out",
                    mfms,
                    *common_vsn,
                    "--seed",
                    args.seed + 4,
                ],
            ),
            (
                "evaluate_cartesian",
                [
                    "scripts/evaluate.py",
                    "--config",
                    "configs/mesh_insertion_tight.yaml",
                    "--method",
                    "all",
                    "--task",
                    "alignment",
                    "--episodes",
                    "1",
                    "--split",
                    "test_unseen",
                    "--sfms-policy",
                    sfms,
                    "--mfms-policy",
                    mfms,
                    "--allow-incompatible",
                    "--out",
                    args.out / "cartesian_evaluation",
                    *common_vsn,
                    "--seed",
                    args.seed + 5,
                ],
            ),
            (
                "evaluate_panda",
                [
                    "scripts/panda_evaluate_controller.py",
                    "--method",
                    "sfss",
                    "--task",
                    "alignment",
                    "--native-camera",
                    "--mask_source",
                    "predicted",
                    "--shapes",
                    "square-square",
                    "--episodes",
                    "1",
                    "--execution-mode",
                    "kinematic",
                    "--out",
                    args.out / "panda_evaluation",
                    *common_vsn[2:],
                    "--seed",
                    args.seed + 6,
                ],
            ),
        ]
    )
    records = []
    for name, raw_command in stages:
        command = [sys.executable, *map(str, raw_command)]
        started = time.perf_counter()
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        records.append(
            {
                "stage": name,
                "command": command,
                "duration_s": time.perf_counter() - started,
                "stdout_tail": completed.stdout[-2000:],
            }
        )
    report = {"passed": True, "seed": args.seed, "stages": records}
    (args.out / "miniature_e2e_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "stages": [row["stage"] for row in records]}, indent=2))


if __name__ == "__main__":
    main()
