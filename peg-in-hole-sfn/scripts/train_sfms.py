#!/usr/bin/env python
import argparse
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
)
from sfn.models.vsn import VirtualSensorNetwork
from sfn.training.train_sfms import (
    SFMSTeacherPretrainConfig,
    SFMSTrainConfig,
    pretrain_sfms_from_sfss_teacher,
    random_policy_smoke,
    train_sfms,
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
    disturbed = DisturbanceConfig(**{**profile.to_dict(), "seed": int(seed)})
    return DisturbedVirtualSensorNetwork(base, disturbed)


def main():
    ap = argparse.ArgumentParser(description="Train the SFMS A2C controller")
    ap.add_argument("--config", default=str(ROOT / "configs" / "sfms.yaml"))
    ap.add_argument("--out", default=str(ROOT / "models" / "sfms.pt"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--rollout-steps", type=int, default=32)
    ap.add_argument("--num-envs", type=int, default=1, help="Number of synchronously stepped training environments.")
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=1e-3)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument(
        "--anchor-imitation-coef",
        type=float,
        default=0.0,
        help="Keep RL fine-tuning near the --init-policy actor; useful for teacher warm-starts.",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Evaluate during training every N updates and optionally save the best checkpoint.",
    )
    ap.add_argument("--eval-episodes", type=int, default=3, help="Episodes per shape for in-training evaluation.")
    ap.add_argument(
        "--eval-split",
        default=None,
        choices=sorted(DEFAULT_SHAPE_SPLITS),
        help="Shape split for in-training evaluation; defaults to training shapes.",
    )
    ap.add_argument("--best-out", default=None, help="Optional path for the best in-training evaluation checkpoint.")
    ap.add_argument("--eval-seed", type=int, default=None, help="Fixed validation seed; defaults to seed+100000.")
    ap.add_argument("--checkpoint-every", type=int, default=0, help="Save an update checkpoint every N updates.")
    ap.add_argument("--log-jsonl", default=None, help="Per-update JSONL metrics path; defaults beside --out.")
    ap.add_argument("--mask_source", default="ground_truth", choices=["ground_truth", "predicted"])
    ap.add_argument("--segmentation", default=None)
    ap.add_argument("--position", default=None)
    ap.add_argument("--orientation", default=None)
    ap.add_argument("--init-policy", default=None, help="Optional SFMS checkpoint to initialize/fine-tune from.")
    ap.add_argument("--resume", default=None, help="Resume model, optimizer, update and global-step state.")
    ap.add_argument(
        "--split", default=None, choices=sorted(DEFAULT_SHAPE_SPLITS), help="Train on this configured shape split."
    )
    ap.add_argument("--shapes", default=None, help="Comma-separated explicit shape list; overrides --split.")
    ap.add_argument("--teacher-pretrain", action="store_true", help="Warm-start SFMS by imitating the SFSS controller.")
    ap.add_argument("--teacher-samples", type=int, default=4096)
    ap.add_argument("--teacher-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--robustness-profile",
        default=None,
        choices=sorted(ROBUSTNESS_PROFILES),
        help="Optional visual disturbance profile to apply during policy-state construction.",
    )
    ap.add_argument("--random-smoke", action="store_true", help="Run only a random-policy rollout smoke test.")
    args = ap.parse_args()

    cfg = load_config(args.config, seed=args.seed)
    shapes = (
        [s.strip() for s in args.shapes.split(",") if s.strip()]
        if args.shapes
        else DEFAULT_SHAPE_SPLITS.get(args.split)
    )
    eval_shapes = DEFAULT_SHAPE_SPLITS.get(args.eval_split) if args.eval_split else None
    vsn = _build_vsn(args, cfg.project.seed)
    if args.random_smoke:
        metrics = random_policy_smoke(
            episodes=3, shapes=shapes, seed=cfg.project.seed, env_config=cfg.environment, camera_config=cfg.camera
        )
    elif args.teacher_pretrain:
        train_cfg = SFMSTeacherPretrainConfig(
            samples=args.teacher_samples,
            epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            seed=cfg.project.seed,
            mask_source=args.mask_source,
            device=cfg.project.device,
        )
        metrics = pretrain_sfms_from_sfss_teacher(
            args.out,
            config=train_cfg,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            segmentation_path=args.segmentation,
            position_path=args.position,
            orientation_path=args.orientation,
            shapes=shapes,
            vsn=vsn,
        )
    else:
        train_cfg = SFMSTrainConfig(
            updates=args.updates,
            rollout_steps=args.rollout_steps,
            num_envs=args.num_envs,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            entropy_coef=args.entropy_coef,
            anchor_imitation_coef=args.anchor_imitation_coef,
            eval_every=args.eval_every,
            eval_episodes_per_shape=args.eval_episodes,
            eval_seed=args.eval_seed,
            checkpoint_every=args.checkpoint_every,
            log_jsonl=args.log_jsonl,
            seed=cfg.project.seed,
            mask_source=args.mask_source,
            device=cfg.project.device,
        )
        metrics = train_sfms(
            args.out,
            config=train_cfg,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            segmentation_path=args.segmentation,
            position_path=args.position,
            orientation_path=args.orientation,
            shapes=shapes,
            vsn=vsn,
            initial_policy_path=args.init_policy,
            resume_path=args.resume,
            eval_shapes=eval_shapes,
            best_out=args.best_out,
        )
    if args.robustness_profile:
        metrics["robustness_profile"] = args.robustness_profile
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
