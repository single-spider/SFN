"""Replay image folders or videos with calibration and optional VSN checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfn.models.vsn import VirtualSensorNetwork
from sfn.sim2real.calibration import load_camera_calibration
from sfn.sim2real.replay import iter_replay, replay_frames


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="image directory or video file")
    parser.add_argument("--output", type=Path, required=True, help="JSONL destination (replaced atomically)")
    parser.add_argument("--calibration", type=Path, help="strict sfn.camera_calibration/v1 JSON")
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--position-checkpoint", type=Path)
    parser.add_argument("--orientation-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-fps", type=float, default=30.0, help="timestamps for image folders")
    parser.add_argument("--min-position-confidence", type=float, default=0.0)
    parser.add_argument("--min-orientation-confidence", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoint_paths = (args.segmentation_checkpoint, args.position_checkpoint, args.orientation_checkpoint)
    vsn = None
    if any(path is not None for path in checkpoint_paths):
        if args.segmentation_checkpoint is None:
            raise SystemExit("--segmentation-checkpoint is required when running RGB VSN inference")
        vsn = VirtualSensorNetwork.from_checkpoints(*checkpoint_paths).to(args.device).eval()
    calibration = load_camera_calibration(args.calibration) if args.calibration else None
    rows = list(
        replay_frames(
            iter_replay(args.source, image_fps=args.image_fps),
            vsn=vsn,
            calibration=calibration,
            device=args.device,
            min_position_confidence=args.min_position_confidence,
            min_orientation_confidence=args.min_orientation_confidence,
            max_frames=None if args.max_frames == 0 else args.max_frames,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text("".join(json.dumps(row.to_dict(), allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(args.output)
    for row in rows:
        print(json.dumps(row.to_dict(), allow_nan=False))
    print(json.dumps({"output": str(args.output), "frames": len(rows), "valid": sum(row.valid for row in rows)}))


if __name__ == "__main__":
    main()
