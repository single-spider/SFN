#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse

from sfn.config import load_config
from sfn.data.collect import collect_npz


def main():
    ap = argparse.ArgumentParser(description="Collect deterministic simulated samples")
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--split", default="train_seen")
    ap.add_argument("--samples-per-shape", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "data" / "smoke"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=None, help="Samples per NPZ chunk; useful for large datasets.")
    ap.add_argument(
        "--include-edge-cases", action="store_true", help="Add boundary/success/workspace pose samples for every shape."
    )
    ap.add_argument(
        "--no-compress", action="store_true", help="Use uncompressed NPZ chunks. Much faster, but larger on disk."
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print collection progress every N samples; 0 disables sample progress.",
    )
    ap.add_argument("--randomization-level", choices=["none", "light", "medium", "heavy"], default="none")
    args = ap.parse_args()
    cfg = load_config(args.config, seed=args.seed)
    print(
        collect_npz(
            args.out,
            args.split,
            args.samples_per_shape,
            cfg.project.seed,
            args.chunk_size,
            args.include_edge_cases,
            not args.no_compress,
            args.progress_every,
            env_config=cfg.environment,
            camera_config=cfg.camera,
            randomization_level=args.randomization_level,
        )
    )


if __name__ == "__main__":
    main()
