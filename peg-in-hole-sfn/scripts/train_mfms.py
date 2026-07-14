#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import load_config
from sfn.constants import DEFAULT_SHAPE_SPLITS
from sfn.evaluation.disturbance import ROBUSTNESS_PROFILES, DisturbanceConfig, DisturbedVirtualSensorNetwork
from sfn.models.vsn import VirtualSensorNetwork
from sfn.training.train_mfms import (
    MFMSTeacherPretrainConfig,
    MFMSTrainConfig,
    pretrain_mfms_from_sfms_teacher,
    pretrain_mfms_from_sfss_teacher,
    train_mfms,
)


def _build_vsn(args, seed: int):
    if not args.robustness_profile:
        return None
    base = VirtualSensorNetwork.from_checkpoints(
        args.segmentation if args.mask_source == "predicted" else None,
        args.position,
        args.orientation,
    )
    profile = ROBUSTNESS_PROFILES[args.robustness_profile]
    disturbance = DisturbanceConfig(**{**profile.to_dict(), "seed": int(seed)})
    return DisturbedVirtualSensorNetwork(base, disturbance)


def main():
    ap = argparse.ArgumentParser(description="Train the MFMS recurrent controller")
    ap.add_argument("--config", default=str(ROOT / "configs" / "mfms.yaml"))
    ap.add_argument("--out", default=str(ROOT / "models" / "mfms.pt"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask_source", default="ground_truth", choices=["ground_truth", "predicted"])
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--position", default=None)
    ap.add_argument("--orientation", default=None)
    ap.add_argument("--split", default=None, choices=sorted(DEFAULT_SHAPE_SPLITS))
    ap.add_argument("--shapes", default=None, help="Comma-separated explicit shape list; overrides --split.")
    ap.add_argument("--teacher-pretrain", action="store_true", help="Warm-start MFMS by imitating recursive SFSS.")
    ap.add_argument("--sfms-teacher", default=None, help="Optional SFMS checkpoint to imitate instead of SFSS.")
    ap.add_argument("--teacher-samples", type=int, default=4096)
    ap.add_argument("--teacher-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--history-len",
        type=int,
        default=None,
        help="History length. Defaults to 4 for pretraining and to the checkpoint metadata for RL fine-tuning.",
    )
    ap.add_argument("--init-policy", default=None, help="Optional MFMS checkpoint to initialize/fine-tune from.")
    ap.add_argument("--resume", default=None, help="Resume model, optimizer, update and global-step state.")
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--burn-in-steps", type=int, default=0, help="Exclude early post-reset rollout steps from losses.")
    ap.add_argument("--actor-lr", type=float, default=3e-5)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.001)
    ap.add_argument("--anchor-imitation-coef", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--eval-seed", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--log-jsonl", default=None)
    ap.add_argument("--eval-split", default=None, choices=sorted(DEFAULT_SHAPE_SPLITS))
    ap.add_argument("--best-out", default=None)
    ap.add_argument(
        "--robustness-profile",
        default=None,
        choices=sorted(ROBUSTNESS_PROFILES),
        help="Optional visual disturbance profile for disturbance-aware MFMS imitation.",
    )
    ap.add_argument(
        "--clean-target",
        action="store_true",
        help="With --robustness-profile and --sfms-teacher, train from disturbed MFMS input but clean SFMS teacher targets.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    shapes = (
        [s.strip() for s in args.shapes.split(",") if s.strip()]
        if args.shapes
        else DEFAULT_SHAPE_SPLITS.get(args.split)
    )
    eval_shapes = DEFAULT_SHAPE_SPLITS.get(args.eval_split) if args.eval_split else None
    vsn = _build_vsn(args, cfg.project.seed)
    target_vsn = None
    if args.clean_target:
        if not args.robustness_profile:
            raise SystemExit("--clean-target is only meaningful with --robustness-profile")
        if not args.sfms_teacher:
            raise SystemExit("--clean-target currently requires --sfms-teacher")
        target_vsn = VirtualSensorNetwork.from_checkpoints(
            args.segmentation if args.mask_source == "predicted" else None,
            args.position,
            args.orientation,
        )
    if args.teacher_pretrain:
        train_cfg = MFMSTeacherPretrainConfig(
            samples=args.teacher_samples,
            epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            history_len=args.history_len or 4,
            seed=cfg.project.seed,
            mask_source=args.mask_source,
            device=cfg.project.device,
        )
        if args.sfms_teacher:
            metrics = pretrain_mfms_from_sfms_teacher(
                args.out,
                sfms_teacher_path=args.sfms_teacher,
                config=train_cfg,
                shapes=shapes,
                env_config=cfg.environment,
                camera_config=cfg.camera,
                segmentation_path=args.segmentation,
                position_path=args.position,
                orientation_path=args.orientation,
                vsn=vsn,
                target_vsn=target_vsn,
            )
        else:
            metrics = pretrain_mfms_from_sfss_teacher(
                args.out,
                config=train_cfg,
                shapes=shapes,
                env_config=cfg.environment,
                camera_config=cfg.camera,
                segmentation_path=args.segmentation,
                position_path=args.position,
                orientation_path=args.orientation,
                vsn=vsn,
            )
    else:
        train_cfg = MFMSTrainConfig(
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            burn_in_steps=args.burn_in_steps,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            entropy_coef=args.entropy_coef,
            anchor_imitation_coef=args.anchor_imitation_coef,
            eval_every=args.eval_every,
            eval_episodes_per_shape=args.eval_episodes,
            eval_seed=args.eval_seed,
            checkpoint_every=args.checkpoint_every,
            log_jsonl=args.log_jsonl,
            history_len=args.history_len or 0,
            seed=cfg.project.seed,
            mask_source=args.mask_source,
            device=cfg.project.device,
        )
        metrics = train_mfms(
            args.out,
            config=train_cfg,
            shapes=shapes,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            segmentation_path=args.segmentation,
            position_path=args.position,
            orientation_path=args.orientation,
            vsn=vsn,
            initial_policy_path=args.init_policy,
            resume_path=args.resume,
            eval_shapes=eval_shapes,
            best_out=args.best_out,
        )
    if args.robustness_profile:
        metrics["robustness_profile"] = args.robustness_profile
    if args.clean_target:
        metrics["clean_target"] = True
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
