"""Validate strict SFN camera calibration JSON and optional recording sizes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfn.sim2real.calibration import load_camera_calibration
from sfn.sim2real.replay import iter_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument(
        "--source", type=Path, help="optional image folder/video to check against calibrated resolution"
    )
    parser.add_argument("--image-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 checks all frames")
    parser.add_argument("--canonical-output", type=Path, help="write normalized, sorted calibration JSON")
    args = parser.parse_args()

    calibration = load_camera_calibration(args.calibration)
    checked = 0
    if args.source is not None:
        expected = (calibration.intrinsics.image_height_px, calibration.intrinsics.image_width_px)
        for frame in iter_replay(args.source, image_fps=args.image_fps):
            if frame.image_rgb.shape[:2] != expected:
                raise SystemExit(
                    f"frame {frame.index} is {frame.image_rgb.shape[1]}x{frame.image_rgb.shape[0]}, "
                    f"expected {expected[1]}x{expected[0]}"
                )
            checked += 1
            if args.max_frames and checked >= args.max_frames:
                break
    if args.canonical_output:
        calibration.to_json(args.canonical_output)

    quaternion_norm = float(np.linalg.norm(calibration.extrinsics.rotation_xyzw))
    summary = {
        "valid": True,
        "schema_version": calibration.schema_version,
        "camera_name": calibration.camera_name,
        "resolution": [calibration.intrinsics.image_width_px, calibration.intrinsics.image_height_px],
        "distortion_model": calibration.distortion.model,
        "transform": f"{calibration.extrinsics.source_frame}->{calibration.extrinsics.target_frame}",
        "transform_direction": calibration.extrinsics.direction,
        "quaternion_norm": quaternion_norm,
        "frames_checked": checked,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
