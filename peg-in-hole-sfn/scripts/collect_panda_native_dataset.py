#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import CameraConfig, EnvironmentConfig
from sfn.constants import DEFAULT_SHAPE_SPLITS
from sfn.data.augment import apply_domain_randomization
from sfn.data.schema import (
    SCHEMA_ID,
    SCHEMA_REVISION,
    SCHEMA_VERSION,
    SPLIT_DEFINITION_VERSION,
    camera_contract,
    canonical_config_hash,
    shape_catalog,
    shape_family,
    source_revision,
    symmetry_order,
)
from sfn.geometry import encode_orientation, encode_position_heatmap, is_success
from sfn.panda.config import PandaConfig
from sfn.panda.panda_scene import PandaScene


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _vec3(value: str) -> tuple[float, float, float]:
    row = tuple(float(v.strip()) for v in value.split(","))
    if len(row) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return row


def _resolution(value: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT") from exc
    return w, h


def _edge_pose_errors() -> list[list[float]]:
    mm = 0.001
    return [
        [0.0, 0.0, 0.0],
        [mm, 0.0, 0.0],
        [-mm, 0.0, 0.0],
        [0.0, mm, 0.0],
        [0.0, -mm, 0.0],
        [10 * mm, 0.0, 0.0],
        [-10 * mm, 0.0, 0.0],
        [0.0, 10 * mm, 0.0],
        [0.0, -10 * mm, 0.0],
        [10 * mm, 10 * mm, 10.0],
        [-10 * mm, -10 * mm, -10.0],
        [10 * mm, -10 * mm, 10.0],
        [-10 * mm, 10 * mm, -10.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, -10.0],
    ]


def _randomize_render(
    rgb: np.ndarray,
    mask: np.ndarray,
    level: str,
    dataset_seed: int,
    sample_id: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the shared deterministic randomization contract to one render."""
    aug_seed = int(
        np.random.SeedSequence([int(dataset_seed), int(sample_id), 0x53464E]).generate_state(
            1, dtype=np.uint32
        )[0]
    )
    return apply_domain_randomization(rgb, mask, level, seed=aug_seed, return_record=True)


def _flush(out_dir: Path, split: str, chunk_idx: int, rows: list[dict], compress: bool) -> dict:
    path = out_dir / f"{split}_{chunk_idx:03d}.npz"
    arrays = {
        "rgb": np.stack([r["rgb"] for r in rows]),
        "mask": np.stack([r["mask"] for r in rows]),
        "pose_error": np.stack([r["pose_error"] for r in rows]),
        "position_target": np.stack([r["position_target"] for r in rows]),
        "orientation_index": np.asarray([r["orientation_index"] for r in rows], dtype=np.int64),
        "shape_id": np.asarray([r["shape_id"] for r in rows]),
        "sample_id": np.asarray([r["sample_id"] for r in rows], dtype=np.int64),
        "seed": np.asarray([r["seed"] for r in rows], dtype=np.int64),
        "camera_variant": np.asarray([r["camera_variant"] for r in rows], dtype=np.int16),
        "augmentation_json": np.asarray([r["augmentation_json"] for r in rows]),
        "shape_family": np.asarray([r["shape_family"] for r in rows]),
        "symmetry_order": np.asarray([r["symmetry_order"] for r in rows], dtype=np.int16),
        "episode_id": np.asarray([r["episode_id"] for r in rows], dtype=np.int64),
        "frame_id": np.asarray([r["frame_id"] for r in rows], dtype=np.int64),
    }
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)
    return {"path": path.name, "sha256": _sha256(path), "samples": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Panda native-camera RGB/body-ID mask dataset.")
    ap.add_argument("--split", default="train_seen", choices=sorted(DEFAULT_SHAPE_SPLITS))
    ap.add_argument("--out", default=str(ROOT / "data" / "panda_native_train_seen"))
    ap.add_argument("--samples-per-shape", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--include-edge-cases", action="store_true")
    ap.add_argument("--no-compress", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--camera-z", type=float, default=None, help="Legacy top-down camera-height override.")
    ap.add_argument("--camera-eye", type=_vec3, default=(0.06, -0.10, 0.16))
    ap.add_argument("--camera-target", type=_vec3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--camera-fov", type=float, default=35.0)
    ap.add_argument("--camera-resolution", type=_resolution, default=(500, 400))
    ap.add_argument("--randomization-level", choices=["none", "light", "medium", "heavy"], default="none")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    env_cfg = EnvironmentConfig()
    camera_cfg = CameraConfig(
        render_width=int(args.camera_resolution[0]),
        render_height=int(args.camera_resolution[1]),
        fov_y_deg=float(args.camera_fov),
        crop_width=int(args.camera_resolution[0]),
        crop_height=int(args.camera_resolution[1]),
    )
    eye = (0.0, 0.0, float(args.camera_z)) if args.camera_z is not None else args.camera_eye
    panda_cfg = PandaConfig(
        native_camera=True,
        mesh_derived_alignment_z=True,
        use_convex_decomposition=False,
        camera_eye_offset_m=eye,
        camera_target_offset_m=args.camera_target,
    )
    shapes = DEFAULT_SHAPE_SPLITS[args.split]
    pending: list[dict] = []
    chunks: list[dict] = []
    sample_id = 0
    chunk_idx = 0
    class_counts = np.zeros(3, dtype=np.int64)
    started = time.time()
    total = len(shapes) * (int(args.samples_per_shape) + (len(_edge_pose_errors()) if args.include_edge_cases else 0))

    def add_sample(scene: PandaScene, pose, shape: str, seed: int):
        nonlocal sample_id, chunk_idx, pending, chunks, class_counts
        scene.reset_to_pose_error(pose)
        rendered = scene.render_camera(camera_cfg)
        rgb, mask, augmentation = _randomize_render(
            rendered.rgb,
            rendered.mask,
            args.randomization_level,
            args.seed,
            sample_id,
        )
        pose_arr = np.asarray(pose, dtype=np.float32)
        row = {
            "rgb": rgb,
            "mask": mask,
            "pose_error": pose_arr,
            "position_target": encode_position_heatmap(float(pose_arr[0]), float(pose_arr[1])),
            "orientation_index": encode_orientation(float(pose_arr[2])),
            "shape_id": shape,
            "sample_id": sample_id,
            "seed": seed,
            "camera_variant": 0,
            "augmentation_json": json.dumps(augmentation, sort_keys=True, separators=(",", ":")),
            "shape_family": shape_family(shape),
            "symmetry_order": symmetry_order(shape),
            "episode_id": sample_id,
            "frame_id": 0,
        }
        pending.append(row)
        class_counts += np.bincount(mask.reshape(-1), minlength=3)[:3]
        sample_id += 1
        if args.progress_every and (sample_id == 1 or sample_id % args.progress_every == 0 or sample_id == total):
            elapsed = max(1e-6, time.time() - started)
            print(
                f"[panda_collect] {sample_id}/{total} ({100 * sample_id / max(1, total):.1f}%, {sample_id / elapsed:.1f} samples/s)",
                flush=True,
            )
        if len(pending) >= int(args.chunk_size):
            meta = _flush(out_dir, args.split, chunk_idx, pending, compress=not args.no_compress)
            chunks.append(meta)
            print(f"[panda_collect] wrote {meta['path']} samples={meta['samples']}", flush=True)
            pending = []
            chunk_idx += 1

    for shape in shapes:
        print(f"[panda_collect] shape={shape}", flush=True)
        with PandaScene(shape=shape, config=panda_cfg, seed=int(args.seed) + sample_id) as scene:
            if args.include_edge_cases:
                for pose in _edge_pose_errors():
                    add_sample(scene, pose, shape, int(args.seed) + sample_id)
            for _ in range(int(args.samples_per_shape)):
                for _attempt in range(1000):
                    pose = np.asarray(
                        [
                            rng.uniform(-env_cfg.xy_initial_range_mm, env_cfg.xy_initial_range_mm) / 1000.0,
                            rng.uniform(-env_cfg.xy_initial_range_mm, env_cfg.xy_initial_range_mm) / 1000.0,
                            rng.uniform(-env_cfg.yaw_initial_range_deg, env_cfg.yaw_initial_range_deg),
                        ],
                        dtype=np.float32,
                    )
                    if not is_success(pose, env_cfg.xy_success_axis_mm, env_cfg.yaw_success_deg):
                        break
                add_sample(scene, pose, shape, int(args.seed) + sample_id)

    if pending:
        meta = _flush(out_dir, args.split, chunk_idx, pending, compress=not args.no_compress)
        chunks.append(meta)
        print(f"[panda_collect] wrote {meta['path']} samples={meta['samples']}", flush=True)

    resolved_config = {
        "environment": asdict(env_cfg),
        "camera": asdict(camera_cfg),
        "panda": panda_cfg.to_dict(),
        "collection": {
            "split": args.split,
            "samples_per_shape": int(args.samples_per_shape),
            "seed": int(args.seed),
            "include_edge_cases": bool(args.include_edge_cases),
            "randomization_level": args.randomization_level,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_revision": SCHEMA_REVISION,
        "schema_id": SCHEMA_ID,
        "metadata_path": "manifest.json",
        "dataset_type": "panda_native_camera_body_id",
        "source_revision": source_revision(ROOT),
        "config_hash": canonical_config_hash(resolved_config),
        "resolved_config": resolved_config,
        "split": args.split,
        "split_definition_version": SPLIT_DEFINITION_VERSION,
        "samples": sample_id,
        "chunks": chunks,
        "seed": int(args.seed),
        "shapes": shapes,
        "shape_catalog": shape_catalog(shapes),
        "modalities": {
            "rgb": {"required": True},
            "mask": {"required": True},
            "depth": {"required": False, "present": False, "units": "m"},
        },
        "randomization": {
            "implementation": "sfn.data.augment",
            "record_version": 1,
            "level": args.randomization_level,
            "per_sample_field": "augmentation_json",
        },
        "class_pixel_counts": {str(i): int(class_counts[i]) for i in range(3)},
        "panda_config": panda_cfg.to_dict(),
        "camera_config": camera_contract(
            camera_cfg,
            model="pinhole",
            eye_m=eye,
            target_m=args.camera_target,
            up=panda_cfg.camera_up_vector,
        ),
        "physics_parameters": {
            "engine": "pybullet",
            "time_step_s": float(panda_cfg.time_step),
            "gravity_m_s2": list(panda_cfg.gravity),
            "peg_mass_kg": float(panda_cfg.peg_mass_kg),
            "peg_lateral_friction": float(panda_cfg.peg_lateral_friction),
            "fixture_lateral_friction": float(panda_cfg.fixture_lateral_friction),
            "collision_margin_m": float(panda_cfg.collision_margin_m),
        },
        "include_edge_cases": bool(args.include_edge_cases),
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out": str(out_dir), "samples": sample_id, "chunks": len(chunks), "elapsed_sec": manifest["elapsed_sec"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
