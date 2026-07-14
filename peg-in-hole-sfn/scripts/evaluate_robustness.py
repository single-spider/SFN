#!/usr/bin/env python
"""Evaluate SFSS/SFMS/MFMS under simple perception disturbances."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import load_config
from sfn.constants import DEFAULT_SHAPE_SPLITS
from sfn.evaluation.disturbance import (
    ROBUSTNESS_PROFILES,
    DisturbanceConfig,
    DisturbedVirtualSensorNetwork,
    EnsembleVirtualSensorNetwork,
    TemporalSmoothedVirtualSensorNetwork,
    parse_profile_names,
)
from sfn.evaluation.evaluate_mfms import evaluate_mfms
from sfn.evaluation.evaluate_sfms import evaluate_sfms
from sfn.evaluation.evaluate_sfss import evaluate_sfss, summarize_episodes, summarize_episodes_by_shape
from sfn.models.vsn import VirtualSensorNetwork


def _write_records(out_dir: Path, records: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if records:
        keys = sorted({k for r in records for k in r})
        with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary_by_shape.json").write_text(
        json.dumps(summarize_episodes_by_shape(records), indent=2) + "\n",
        encoding="utf-8",
    )


def _write_combined(out_dir: Path, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with (out_dir / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "robustness_summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Robustness evaluation report",
        "",
        "This is a first-pass disturbance check. It keeps the same controller and task,",
        "but corrupts the visual input before position/orientation inference.",
        "",
        "| Profile | Method | Success rate | Mean steps | Final XY mm | Final yaw deg | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['profile']} | {r['method']} | {r['success_rate']:.3f} | "
            f"{r.get('mean_steps', 0.0):.3f} | {r.get('mean_final_xy_error_mm', 0.0):.3f} | "
            f"{r.get('mean_final_yaw_error_deg', 0.0):.3f} | {r.get('notes', '')} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_names(text: str) -> list[str]:
    out = [x.strip() for x in text.split(",") if x.strip()]
    allowed = {"sfss", "sfms", "mfms"}
    bad = [x for x in out if x not in allowed]
    if bad:
        raise ValueError(f"Unknown method(s): {', '.join(bad)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate controllers under simple visual disturbances.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "sfms_strict_xy.yaml"))
    ap.add_argument("--methods", default="sfms,mfms", help="Comma-separated subset of sfss,sfms,mfms.")
    ap.add_argument("--profiles", default=None, help="Comma-separated robustness profiles. Defaults to all built-ins.")
    ap.add_argument("--split", default="test_unseen", choices=sorted(DEFAULT_SHAPE_SPLITS))
    ap.add_argument("--shapes", default=None, help="Comma-separated explicit shape list; overrides --split.")
    ap.add_argument("--task", default="insertion", choices=["alignment", "insertion"])
    ap.add_argument("--mask_source", default="predicted", choices=["ground_truth", "predicted"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1200)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "robustness_eval"))
    ap.add_argument("--segmentation", default=str(ROOT / "models" / "segmentation.pt"))
    ap.add_argument("--position", default=str(ROOT / "models" / "position.pt"))
    ap.add_argument("--orientation", default=str(ROOT / "models" / "orientation.pt"))
    ap.add_argument("--sfms-policy", default=str(ROOT / "models" / "sfms_strict_xy_anchor1_best.pt"))
    ap.add_argument(
        "--mfms-policy", default=str(ROOT / "models" / "mfms_gt_sfms_strict_xy_teacher_train_seen_4096_e30.pt")
    )
    ap.add_argument("--confidence-mode", default="scale", choices=["ignore", "scale", "hold"])
    ap.add_argument(
        "--ensemble-samples", type=int, default=1, help="Average this many disturbed VSN samples per observation."
    )
    ap.add_argument(
        "--temporal-alpha",
        type=float,
        default=None,
        help="Optional EMA weight for smoothing VSN output probabilities over time.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    shapes = (
        [s.strip() for s in args.shapes.split(",") if s.strip()] if args.shapes else DEFAULT_SHAPE_SPLITS[args.split]
    )
    methods = _method_names(args.methods)
    profile_names = parse_profile_names(args.profiles)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    base_vsn = VirtualSensorNetwork.from_checkpoints(
        args.segmentation if args.mask_source == "predicted" else None,
        args.position,
        args.orientation,
    )

    rows: list[dict] = []
    for profile_name in profile_names:
        base_profile = ROBUSTNESS_PROFILES[profile_name]
        profile = DisturbanceConfig(**{**base_profile.to_dict(), "seed": int(args.seed) + len(rows) * 17})
        for method in methods:
            vsn = DisturbedVirtualSensorNetwork(base_vsn, profile)
            if args.ensemble_samples > 1:
                vsn = EnsembleVirtualSensorNetwork(vsn, samples=args.ensemble_samples)
            if args.temporal_alpha is not None:
                vsn = TemporalSmoothedVirtualSensorNetwork(vsn, alpha=args.temporal_alpha)
            suffix = method
            if args.ensemble_samples > 1:
                suffix += f"_ens{args.ensemble_samples}"
            if args.temporal_alpha is not None:
                suffix += f"_ema{args.temporal_alpha:g}"
            method_out = out_root / profile.name / suffix
            if method == "sfss":
                records, _steps = evaluate_sfss(
                    segmentation_path=args.segmentation,
                    position_path=args.position,
                    orientation_path=args.orientation,
                    shapes=shapes,
                    episodes_per_shape=args.episodes,
                    seed=cfg.project.seed,
                    task=args.task,
                    mask_source=args.mask_source,
                    recursive=True,
                    env_config=cfg.environment,
                    insertion_config=cfg.insertion,
                    confidence_mode=args.confidence_mode,
                    vsn=vsn,
                )
            elif method == "sfms":
                records = evaluate_sfms(
                    policy_path=args.sfms_policy,
                    shapes=shapes,
                    episodes_per_shape=args.episodes,
                    seed=cfg.project.seed,
                    mask_source=args.mask_source,
                    task=args.task,
                    env_config=cfg.environment,
                    insertion_config=cfg.insertion,
                    vsn=vsn,
                    device=cfg.project.device,
                )
            else:
                records = evaluate_mfms(
                    policy_path=args.mfms_policy,
                    shapes=shapes,
                    episodes_per_shape=args.episodes,
                    seed=cfg.project.seed,
                    mask_source=args.mask_source,
                    task=args.task,
                    env_config=cfg.environment,
                    insertion_config=cfg.insertion,
                    vsn=vsn,
                    device=cfg.project.device,
                )
            summary = summarize_episodes(records)
            summary.update(
                {
                    "profile": profile.name,
                    "method": method,
                    "task": args.task,
                    "mask_source": args.mask_source,
                    "episodes_per_shape": int(args.episodes),
                    "split": args.split,
                    "disturbance": profile.to_dict(),
                    "ensemble_samples": int(args.ensemble_samples),
                    "temporal_alpha": args.temporal_alpha,
                    "notes": (
                        "visual disturbance plus VSN probability ensembling/smoothing"
                        if args.temporal_alpha is not None or args.ensemble_samples > 1
                        else "visual disturbance before VSN position/orientation inference"
                    ),
                }
            )
            _write_records(method_out, records, summary)
            rows.append(summary)

    _write_combined(out_root, rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
