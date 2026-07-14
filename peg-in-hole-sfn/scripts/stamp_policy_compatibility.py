#!/usr/bin/env python
"""Create a policy copy with explicit VSN and renderer compatibility metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from sfn.training.common import file_sha256, load_checkpoint_cpu, save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--renderer-backend", required=True)
    parser.add_argument("--mask-source", choices=("ground_truth", "predicted"), required=True)
    parser.add_argument("--segmentation", type=Path, default=None)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--orientation", type=Path, required=True)
    args = parser.parse_args()
    if args.out.resolve() == args.policy.resolve():
        raise SystemExit("--out must differ from --policy so the historical checkpoint remains immutable")
    checkpoint = load_checkpoint_cpu(args.policy)
    checkpoint["compatibility"] = {
        "renderer_backend": args.renderer_backend,
        "mask_source": args.mask_source,
        "segmentation_sha256": file_sha256(args.segmentation) if args.mask_source == "predicted" else None,
        "position_sha256": file_sha256(args.position),
        "orientation_sha256": file_sha256(args.orientation),
    }
    checkpoint.setdefault("provenance", {})["compatibility_stamped_from"] = {
        "path": str(args.policy),
        "sha256": file_sha256(args.policy),
    }
    save_checkpoint(args.out, checkpoint)
    print(args.out)


if __name__ == "__main__":
    main()
