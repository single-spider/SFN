#!/usr/bin/env python
"""Sweep Panda native-camera candidates over real shapes and pose grids."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.constants import ALL_EXPECTED_SHAPES  # noqa: E402
from sfn.panda.camera_observability import (  # noqa: E402
    CameraCandidate,
    SweepThresholds,
    sweep_camera_observability,
)
from sfn.panda.config import PandaConfig  # noqa: E402


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _vectors(value: str) -> list[tuple[float, float, float]]:
    result = []
    for item in value.split(";"):
        row = tuple(_floats(item))
        if len(row) != 3:
            raise argparse.ArgumentTypeError(f"expected x,y,z vector, got {item!r}")
        result.append(row)
    return result


def _resolutions(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        try:
            width, height = (int(v) for v in item.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {item!r}") from exc
        result.append((width, height))
    return result


def _shape_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_EXPECTED_SHAPES)
    shapes = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(shapes) - set(ALL_EXPECTED_SHAPES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown shapes: {unknown}")
    return shapes


def build_candidates(args: argparse.Namespace) -> list[CameraCandidate]:
    candidates = []
    for index, (eye, target, fov, resolution) in enumerate(
        product(args.eye_offsets, args.target_offsets, args.fov, args.resolutions)
    ):
        width, height = resolution
        candidates.append(
            CameraCandidate(
                eye_offset_m=eye,
                target_offset_m=target,
                fov_y_deg=fov,
                width=width,
                height=height,
                near=args.near,
                far=args.far,
                name=f"camera_{index:03d}",
            )
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep eye/target/FOV/resolution over native PyBullet body-ID masks, "
            "including clipping, pixel sensitivity, and symmetry-aware yaw diagnostics."
        )
    )
    parser.add_argument("--shapes", type=_shape_list, default=list(ALL_EXPECTED_SHAPES))
    parser.add_argument(
        "--eye-offsets",
        type=_vectors,
        default=_vectors("0,0,0.10;0.04,-0.04,0.12;-0.04,-0.04,0.12"),
        help='Semicolon-separated task-frame vectors, e.g. "0,0,.10;.04,-.04,.12".',
    )
    parser.add_argument(
        "--target-offsets",
        type=_vectors,
        default=_vectors("0,0,0;0,0,0.02"),
        help="Semicolon-separated task-frame look-at offsets.",
    )
    parser.add_argument("--fov", type=_floats, default=[35.0, 45.0])
    parser.add_argument("--resolutions", type=_resolutions, default=[(250, 200)])
    parser.add_argument("--near", type=float, default=0.001)
    parser.add_argument("--far", type=float, default=1.0)
    parser.add_argument("--grid-mm", type=_floats, default=[-10.0, 0.0, 10.0])
    parser.add_argument("--grid-yaw-deg", type=_floats, default=[-10.0, 0.0, 10.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "panda_camera_observability.json")
    parser.add_argument("--min-peg-pixels", type=int, default=20)
    parser.add_argument("--min-seam-pixels", type=int, default=5)
    parser.add_argument("--min-visible-fraction", type=float, default=0.95)
    parser.add_argument("--min-unclipped-fraction", type=float, default=0.95)
    parser.add_argument("--min-xy-px-per-mm", type=float, default=0.20)
    parser.add_argument("--min-yaw-change-per-deg", type=float, default=0.0005)
    args = parser.parse_args()

    candidates = build_candidates(args)
    poses = [
        (x_mm / 1000.0, y_mm / 1000.0, yaw_deg)
        for x_mm, y_mm, yaw_deg in product(args.grid_mm, args.grid_mm, args.grid_yaw_deg)
    ]
    thresholds = SweepThresholds(
        min_peg_pixels=args.min_peg_pixels,
        min_seam_pixels=args.min_seam_pixels,
        min_visible_fraction=args.min_visible_fraction,
        min_unclipped_fraction=args.min_unclipped_fraction,
        min_xy_sensitivity_px_per_mm=args.min_xy_px_per_mm,
        min_yaw_change_per_deg=args.min_yaw_change_per_deg,
    )
    panda_config = PandaConfig(
        native_camera=True,
        mesh_derived_alignment_z=True,
        use_convex_decomposition=False,
    )
    print(
        f"[panda_camera] shapes={len(args.shapes)} poses={len(poses)} "
        f"candidates={len(candidates)} renders={len(args.shapes) * len(poses) * len(candidates)}",
        flush=True,
    )
    report = sweep_camera_observability(
        shapes=args.shapes,
        poses=poses,
        candidates=candidates,
        panda_config=panda_config,
        thresholds=thresholds,
        seed=args.seed,
    )
    report["command"] = {
        "candidate_count": len(candidates),
        "pose_count": len(poses),
        "panda_config": panda_config.to_dict(),
        "thresholds": asdict(thresholds),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {
        "out": str(args.out.resolve()),
        "recommended_candidate_id": report["recommended_candidate_id"],
        "viable_candidate_ids": report["viable_candidate_ids"],
        "ranking": [
            {
                "candidate_id": row["candidate"]["candidate_id"],
                "viable": row["viable"],
                "rejection_reasons": row["rejection_reasons"],
                "visible_fraction": row["visible_fraction"],
                "unclipped_fraction": row["unclipped_fraction"],
                "minimum_px_per_mm": row["xy_sensitivity"]["minimum_px_per_mm"],
            }
            for row in report["candidates"]
        ],
    }
    print(json.dumps(compact, indent=2))
    if not report["viable_candidate_ids"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
