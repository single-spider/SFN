from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from .perception import hyperparameter_search, train_perception


def add_perception_args(ap: argparse.ArgumentParser, *, root: Path, task: str, default_config: str) -> None:
    ap.add_argument("--config", default=str(root / "configs" / default_config))
    ap.add_argument("--dataset", default=str(root / "data" / "smoke"))
    ap.add_argument("--val-dataset", default=None)
    ap.add_argument("--out", default=str(root / "models" / f"{task}.pt"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--limit-strategy",
        default="stratified",
        choices=["prefix", "shuffle", "stratified"],
        help="How --limit/--val-limit subsets are selected. Use stratified for research runs.",
    )
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu, cuda, or auto. Defaults to config project.device.")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--base-channels", type=int, default=16)
    ap.add_argument(
        "--orientation-architecture",
        default="relative",
        choices=["relative", "legacy"],
        help="Orientation model: shared Siamese peg/seam correlation (default) or historical joint-mask model.",
    )
    ap.add_argument("--loss", default="auto", choices=["auto", "ce", "weighted_ce", "focal"])
    ap.add_argument("--class-weight", default="median", choices=["none", "median", "inverse", "effective"])
    ap.add_argument("--position-pos-weight", type=float, default=25.0)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--resume", default=None, help="Resume from a .pt checkpoint, usually *.last.pt")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--checkpoint-every", type=int, default=1)
    ap.add_argument("--keep-epoch-checkpoints", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--search-grid", default=None, help='JSON or "lr=1e-3,3e-4;loss=weighted_ce,focal"')
    ap.add_argument("--max-trials", type=int, default=None)


def run_perception_cli(task: str, args) -> dict:
    cfg = load_config(args.config, seed=args.seed)
    device = args.device or cfg.project.device
    common = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit=args.limit,
        limit_strategy=args.limit_strategy,
        val_dataset=args.val_dataset,
        val_fraction=args.val_fraction,
        val_limit=args.val_limit,
        device=device,
        num_workers=args.num_workers,
        base_channels=args.base_channels,
        weight_decay=args.weight_decay,
        loss=args.loss,
        class_weight=args.class_weight,
        position_pos_weight=args.position_pos_weight,
        focal_gamma=args.focal_gamma,
        resume=args.resume,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_every=args.checkpoint_every,
        keep_epoch_checkpoints=args.keep_epoch_checkpoints,
        amp=args.amp,
        progress=not args.no_progress,
        metric=args.metric,
        orientation_architecture=args.orientation_architecture,
    )
    if args.search_grid:
        return hyperparameter_search(
            task,
            args.dataset,
            Path(args.out).with_suffix(""),
            args.search_grid,
            max_trials=args.max_trials,
            seed=cfg.project.seed,
            **common,
        )
    return train_perception(task, args.dataset, args.out, seed=cfg.project.seed, **common)


def print_result(result: dict) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
