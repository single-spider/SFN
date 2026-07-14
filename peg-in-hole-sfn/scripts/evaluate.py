#!/usr/bin/env python
import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import load_config
from sfn.constants import DEFAULT_SHAPE_SPLITS
from sfn.envs.asset_registry import AssetRegistry
from sfn.evaluation.artifacts import write_records_csv
from sfn.evaluation.disturbance import (
    ROBUSTNESS_PROFILES,
    DisturbanceConfig,
    DisturbedVirtualSensorNetwork,
    EnsembleVirtualSensorNetwork,
    TemporalSmoothedVirtualSensorNetwork,
)
from sfn.evaluation.evaluate_contract import grouped_summary
from sfn.evaluation.evaluate_mfms import evaluate_mfms
from sfn.evaluation.evaluate_oracle import evaluate_oracle
from sfn.evaluation.evaluate_random import evaluate_random
from sfn.evaluation.evaluate_sfms import evaluate_sfms
from sfn.evaluation.evaluate_sfss import evaluate_sfss
from sfn.evaluation.provenance import write_run_provenance
from sfn.models.vsn import VirtualSensorNetwork
from sfn.panda.artifacts import write_panda_per_shape


def _write_episode_outputs(records: list[dict], steps: list[dict], out_dir: Path, *, episode_budget: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(out_dir / "episodes.csv", records)
    write_records_csv(out_dir / "steps.csv", steps)
    write_panda_per_shape(out_dir / "per_shape.csv", records)
    methods = grouped_summary(records, "method")
    summary = {
        "episode_budget_per_method": int(episode_budget),
        "total_episode_records": len(records),
        "total_step_records": len(steps),
        "methods": methods,
    }
    if len(methods) == 1:
        summary.update(next(iter(methods.values())))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    per_shape: dict[str, dict] = {}
    for shape in sorted({str(record.get("shape", "unknown")) for record in records}):
        shape_records = [record for record in records if str(record.get("shape", "unknown")) == shape]
        per_shape[shape] = grouped_summary(shape_records, "method")
    (out_dir / "summary_by_shape.json").write_text(json.dumps(per_shape, indent=2) + "\n", encoding="utf-8")
    return summary


def _build_vsn(args, seed: int):
    if args.ensemble_samples <= 1 and args.robustness_profile is None and args.temporal_alpha is None:
        return None
    base = VirtualSensorNetwork.from_checkpoints(
        args.segmentation if args.mask_source == "predicted" else None,
        args.position,
        args.orientation,
    )
    vsn = base
    if args.robustness_profile is not None:
        profile = ROBUSTNESS_PROFILES[args.robustness_profile]
        disturbance = DisturbanceConfig(**{**profile.to_dict(), "seed": int(seed)})
        vsn = DisturbedVirtualSensorNetwork(vsn, disturbance)
    if args.ensemble_samples > 1:
        vsn = EnsembleVirtualSensorNetwork(vsn, samples=args.ensemble_samples)
    if args.temporal_alpha is not None:
        vsn = TemporalSmoothedVirtualSensorNetwork(vsn, alpha=args.temporal_alpha)
    return vsn


def _read_typed_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in list(row.items()):
            if value in {"True", "False"}:
                row[key] = value == "True"
            elif value in {"", "None"}:
                row[key] = None
            else:
                try:
                    row[key] = float(value) if any(char in value for char in ".eE") else int(value)
                except (TypeError, ValueError):
                    pass
    return rows


def _run_panda_backend(args, out: Path, seed: int) -> dict:
    methods = ("oracle", "sfss", "sfms", "mfms") if args.method == "all" else (args.method,)
    if "random" in methods:
        raise SystemExit("Panda backend does not expose the diagnostic random controller")
    all_records, all_steps = [], []
    execution_mode = "kinematic" if args.backend == "panda_kinematic" else "dynamic"
    for method in methods:
        destination = out / method
        command = [
            sys.executable,
            "scripts/panda_evaluate_controller.py",
            "--method",
            method,
            "--task",
            args.task,
            "--mask_source",
            args.mask_source,
            "--episodes",
            str(args.episodes),
            "--seed",
            str(seed),
            "--execution-mode",
            execution_mode,
            "--out",
            str(destination),
        ]
        if args.split:
            command.extend(["--split", args.split])
        if args.shapes:
            command.extend(["--shapes", args.shapes])
        if args.native_camera:
            command.append("--native-camera")
        if args.native_template_vsn:
            command.append("--native-template-vsn")
        if method == "sfms":
            command.extend(["--policy", args.sfms_policy or args.policy])
        elif method == "mfms":
            command.extend(["--policy", args.mfms_policy or args.policy])
        if args.segmentation:
            command.extend(["--segmentation", args.segmentation])
        if args.position:
            command.extend(["--position", args.position])
        if args.orientation:
            command.extend(["--orientation", args.orientation])
        if args.robustness_profile:
            command.extend(["--robustness-profile", args.robustness_profile])
        subprocess.run(command, cwd=ROOT, check=True)
        all_records.extend(_read_typed_csv(destination / "episodes.csv"))
        all_steps.extend(_read_typed_csv(destination / "steps.csv"))
    summary = _write_episode_outputs(all_records, all_steps, out, episode_budget=args.episodes)
    write_panda_per_shape(out / "per_shape.csv", all_records)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Evaluate SFN controllers")
    ap.add_argument("--config", default=str(ROOT / "configs" / "evaluation.yaml"))
    ap.add_argument(
        "--backend",
        default="cartesian",
        choices=["cartesian", "panda_kinematic", "panda_dynamic"],
        help="Execution backend; Cartesian renderer is selected by --config.",
    )
    ap.add_argument("--method", default="oracle", choices=["oracle", "random", "sfss", "sfms", "mfms", "all"])
    ap.add_argument("--mask_source", default="ground_truth", choices=["ground_truth", "predicted"])
    ap.add_argument("--task", default="alignment", choices=["alignment", "insertion"])
    ap.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Total episodes per method, balanced across selected shapes (not episodes per shape).",
    )
    ap.add_argument(
        "--split", default=None, choices=sorted(DEFAULT_SHAPE_SPLITS), help="Evaluate only this configured shape split."
    )
    ap.add_argument("--shapes", default=None, help="Comma-separated explicit shape list; overrides --split.")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "evaluation"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--position", default=None)
    ap.add_argument("--orientation", default=None)
    ap.add_argument(
        "--policy",
        default=None,
        help="Compatibility checkpoint used for SFMS and/or MFMS unless a method-specific option is set.",
    )
    ap.add_argument("--sfms-policy", default=None, help="SFMS policy checkpoint (overrides --policy for SFMS).")
    ap.add_argument("--mfms-policy", default=None, help="MFMS policy checkpoint (overrides --policy for MFMS).")
    ap.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="Diagnostic override for missing/mismatched policy-VSN/backend metadata.",
    )
    ap.add_argument("--one-step", action="store_true", help="Run one-step SFSS instead of recursive SFSS.")
    ap.add_argument("--confidence-mode", default="scale", choices=["ignore", "scale", "hold"])
    ap.add_argument("--sfss-gain-xy", type=float, default=0.7)
    ap.add_argument("--sfss-gain-yaw", type=float, default=0.7)
    ap.add_argument("--save-visuals", action="store_true")
    ap.add_argument("--native-camera", action="store_true", help="Use native PyBullet camera on Panda backends.")
    ap.add_argument("--native-template-vsn", action="store_true", help="Use Panda top-down template pose decoder.")
    ap.add_argument(
        "--robustness-profile",
        default=None,
        choices=sorted(ROBUSTNESS_PROFILES),
        help="Optional disturbance profile for normal eval.",
    )
    ap.add_argument(
        "--ensemble-samples", type=int, default=1, help="Average this many VSN probability samples per observation."
    )
    ap.add_argument(
        "--temporal-alpha", type=float, default=None, help="Optional EMA weight for VSN probability smoothing."
    )
    args = ap.parse_args()

    if args.episodes < 0:
        ap.error("--episodes must be non-negative")
    sfms_policy = args.sfms_policy or args.policy
    mfms_policy = args.mfms_policy or args.policy
    if args.method in {"sfms", "all"} and not sfms_policy:
        ap.error("--sfms-policy (or compatibility --policy) is required for this method")
    if args.method in {"mfms", "all"} and not mfms_policy:
        ap.error("--mfms-policy (or compatibility --policy) is required for this method")

    cfg = load_config(args.config, seed=args.seed)
    out = Path(args.out)
    shapes = (
        [s.strip() for s in args.shapes.split(",") if s.strip()]
        if args.shapes
        else list(DEFAULT_SHAPE_SPLITS[args.split])
        if args.split
        else (AssetRegistry().list_shapes() or ["synthetic-square"])
    )
    manifest = write_run_provenance(
        out,
        resolved_config=asdict(cfg),
        arguments=vars(args),
        input_paths={
            "config": args.config,
            "segmentation": args.segmentation,
            "position": args.position,
            "orientation": args.orientation,
            "sfms_policy": sfms_policy,
            "mfms_policy": mfms_policy,
        },
        seed=cfg.project.seed,
        backend=args.backend if args.backend != "cartesian" else cfg.camera.renderer_backend,
    )
    if args.backend != "cartesian":
        summary = _run_panda_backend(args, out, cfg.project.seed)
        print(json.dumps(summary, indent=2))
        return
    vsn = _build_vsn(args, cfg.project.seed)
    all_records: list[dict] = []
    all_steps: list[dict] = []
    if args.method == "random":
        records, steps = evaluate_random(
            shapes=shapes,
            episodes=args.episodes,
            seed=cfg.project.seed,
            task=args.task,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            insertion_config=cfg.insertion,
        )
        all_records.extend(records)
        all_steps.extend(steps)
    if args.method in {"oracle", "all"}:
        records, steps = evaluate_oracle(
            shapes=shapes,
            episodes=args.episodes,
            seed=cfg.project.seed,
            task=args.task,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            insertion_config=cfg.insertion,
        )
        all_records.extend(records)
        all_steps.extend(steps)

    if args.method in {"sfss", "all"}:
        recursive_modes = (False, True) if args.method == "all" else (not args.one_step,)
        for recursive in recursive_modes:
            records, steps = evaluate_sfss(
                segmentation_path=args.segmentation,
                position_path=args.position,
                orientation_path=args.orientation,
                shapes=shapes,
                episodes=args.episodes,
                seed=cfg.project.seed,
                task=args.task,
                mask_source=args.mask_source,
                recursive=recursive,
                env_config=cfg.environment,
                camera_config=cfg.camera,
                insertion_config=cfg.insertion,
                confidence_mode=args.confidence_mode,
                gain_xy=args.sfss_gain_xy,
                gain_yaw=args.sfss_gain_yaw,
                save_visuals=args.save_visuals,
                visual_dir=out / "visuals" / ("recursive" if recursive else "one_step"),
                vsn=vsn,
            )
            all_records.extend(records)
            all_steps.extend(steps)

    if args.method in {"sfms", "all"}:
        records, steps = evaluate_sfms(
            policy_path=sfms_policy,
            segmentation_path=args.segmentation,
            position_path=args.position,
            orientation_path=args.orientation,
            shapes=shapes,
            episodes=args.episodes,
            seed=cfg.project.seed,
            mask_source=args.mask_source,
            task=args.task,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            insertion_config=cfg.insertion,
            vsn=vsn,
            device=cfg.project.device,
            return_steps=True,
            enforce_compatibility=True,
            allow_incompatible=args.allow_incompatible,
        )
        all_records.extend(records)
        all_steps.extend(steps)

    if args.method in {"mfms", "all"}:
        records, steps = evaluate_mfms(
            policy_path=mfms_policy,
            segmentation_path=args.segmentation,
            position_path=args.position,
            orientation_path=args.orientation,
            shapes=shapes,
            episodes=args.episodes,
            seed=cfg.project.seed,
            mask_source=args.mask_source,
            task=args.task,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            insertion_config=cfg.insertion,
            vsn=vsn,
            device=cfg.project.device,
            return_steps=True,
            enforce_compatibility=True,
            allow_incompatible=args.allow_incompatible,
        )
        all_records.extend(records)
        all_steps.extend(steps)

    common = {
        "split": args.split,
        "mask_source": args.mask_source,
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "segmentation_sha256": (manifest["inputs"].get("segmentation") or {}).get("sha256"),
        "position_sha256": (manifest["inputs"].get("position") or {}).get("sha256"),
        "orientation_sha256": (manifest["inputs"].get("orientation") or {}).get("sha256"),
    }
    for record in all_records:
        record.update(common)
        policy_key = "mfms_policy" if record.get("method") == "mfms" else "sfms_policy"
        record["policy_sha256"] = (manifest["inputs"].get(policy_key) or {}).get("sha256")
    for step in all_steps:
        step.update(common)
    summary = _write_episode_outputs(all_records, all_steps, out, episode_budget=args.episodes)
    if args.ensemble_samples > 1 or args.robustness_profile is not None or args.temporal_alpha is not None:
        summary.update(
            {
                "robustness_profile": args.robustness_profile,
                "ensemble_samples": int(args.ensemble_samples),
                "temporal_alpha": args.temporal_alpha,
            }
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
