from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import CameraConfig, EnvironmentConfig
from ..envs import PegInHoleAlignmentEnv
from ..geometry import encode_orientation, encode_position_heatmap
from .augment import apply_domain_randomization
from .schema import (
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
from .splits import get_split


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _edge_pose_errors() -> list[list[float]]:
    mm = 0.001
    return [
        [0.0, 0.0, 0.0],
        [1 * mm, 0.0, 0.0],
        [-1 * mm, 0.0, 0.0],
        [0.0, 1 * mm, 0.0],
        [0.0, -1 * mm, 0.0],
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


def _flush_chunk(
    out_dir: Path,
    split: str,
    chunk_index: int,
    rgbs: list[np.ndarray],
    masks: list[np.ndarray],
    poses: list[np.ndarray],
    pos_targets: list[np.ndarray],
    ori_idx: list[int],
    shape_ids: list[str],
    sample_ids: list[int],
    seeds: list[int],
    camera_variants: list[int],
    augmentation_json: list[str],
    shape_families: list[str],
    symmetry_orders: list[int],
    episode_ids: list[int],
    frame_ids: list[int],
    depths: list[np.ndarray] | None,
    compress: bool = True,
) -> dict:
    chunk = out_dir / f"{split}_{chunk_index:03d}.npz"
    rgb = np.stack(rgbs)
    mask = np.stack(masks)
    pose_error = np.stack(poses)
    position_target = np.stack(pos_targets)
    arrays = dict(
        rgb=rgb,
        mask=mask,
        pose_error=pose_error,
        position_target=position_target,
        orientation_index=np.asarray(ori_idx, dtype=np.int64),
        shape_id=np.asarray(shape_ids),
        sample_id=np.asarray(sample_ids, dtype=np.int64),
        seed=np.asarray(seeds, dtype=np.int64),
        camera_variant=np.asarray(camera_variants, dtype=np.int16),
        augmentation_json=np.asarray(augmentation_json),
        shape_family=np.asarray(shape_families),
        symmetry_order=np.asarray(symmetry_orders, dtype=np.int16),
        episode_id=np.asarray(episode_ids, dtype=np.int64),
        frame_id=np.asarray(frame_ids, dtype=np.int64),
    )
    if depths is not None:
        arrays["depth"] = np.stack(depths).astype(np.float32)
    if compress:
        np.savez_compressed(chunk, **arrays)
    else:
        np.savez(chunk, **arrays)
    return {"path": chunk.name, "sha256": _sha256(chunk), "samples": len(rgbs)}


def collect_npz(
    out_dir: str | Path,
    split: str = "train_seen",
    samples_per_shape: int = 32,
    seed: int = 1,
    chunk_size: int | None = None,
    include_edge_cases: bool = False,
    compress: bool = True,
    progress_every: int = 100,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    randomization_level: str = "none",
) -> Path:
    """Collect deterministic simulated samples into chunked compressed NPZ files."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(samples_per_shape) < 0:
        raise ValueError("samples_per_shape must be non-negative")
    shapes = get_split(split)
    chunk_size = int(chunk_size or max(1, samples_per_shape * max(1, len(shapes))))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    rgbs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    pos_targets: list[np.ndarray] = []
    ori_idx: list[int] = []
    shape_ids: list[str] = []
    sample_ids: list[int] = []
    seeds: list[int] = []
    camera_variants: list[int] = []
    augmentation_json: list[str] = []
    shape_families: list[str] = []
    symmetry_orders: list[int] = []
    episode_ids: list[int] = []
    frame_ids: list[int] = []
    depths: list[np.ndarray] = []
    depth_available: bool | None = None
    chunks: list[dict] = []
    class_counts = np.zeros(3, dtype=np.int64)
    pose_min = np.asarray([np.inf, np.inf, np.inf], dtype=np.float64)
    pose_max = np.asarray([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    sample_id = 0
    chunk_index = 0
    first_chunk: Path | None = None
    total_target = len(shapes) * (int(samples_per_shape) + (len(_edge_pose_errors()) if include_edge_cases else 0))
    started = time.time()

    def report(force: bool = False) -> None:
        if not force and (progress_every <= 0 or sample_id % progress_every != 0):
            return
        elapsed = max(1e-6, time.time() - started)
        rate = sample_id / elapsed
        pct = 100.0 * sample_id / max(1, total_target)
        print(
            f"[collect_dataset] {sample_id}/{total_target} samples "
            f"({pct:.1f}%, {rate:.1f} samples/s, chunks={len(chunks)})",
            flush=True,
        )

    def add_sample(obs: dict, shape: str, sample_seed: int) -> None:
        nonlocal sample_id, chunk_index, first_chunk, class_counts, pose_min, pose_max, depth_available
        pose = obs["pose_error"]
        aug_seed = int(np.random.SeedSequence([seed, sample_id, 0x53464E]).generate_state(1, dtype=np.uint32)[0])
        rgb, mask, aug = apply_domain_randomization(
            obs["rgb"], obs["mask"], randomization_level, seed=aug_seed, return_record=True
        )
        rgbs.append(rgb)
        masks.append(mask)
        poses.append(pose)
        pos_targets.append(encode_position_heatmap(float(pose[0]), float(pose[1])))
        ori_idx.append(encode_orientation(float(pose[2])))
        shape_ids.append(shape)
        sample_ids.append(sample_id)
        seeds.append(sample_seed)
        camera_variants.append(0)
        augmentation_json.append(json.dumps(aug, sort_keys=True, separators=(",", ":")))
        shape_families.append(shape_family(shape))
        symmetry_orders.append(symmetry_order(shape))
        # Collection resets once per observation, so every sample is a
        # one-frame episode.  Sequential collectors can use non-zero frame IDs.
        episode_ids.append(sample_id)
        frame_ids.append(0)
        has_depth = obs.get("depth") is not None
        if depth_available is None:
            depth_available = has_depth
        if has_depth != depth_available:
            raise ValueError("depth must be present for every sample or omitted for the whole dataset")
        if has_depth:
            depths.append(np.asarray(obs["depth"], dtype=np.float32))
        class_counts += np.bincount(mask.reshape(-1), minlength=3)[:3]
        pose_arr = np.asarray(pose, dtype=np.float64)
        pose_min = np.minimum(pose_min, pose_arr)
        pose_max = np.maximum(pose_max, pose_arr)
        sample_id += 1
        report()
        if len(rgbs) >= chunk_size:
            print(f"[collect_dataset] writing chunk {chunk_index:03d} ({len(rgbs)} samples)...", flush=True)
            meta = _flush_chunk(
                out_dir,
                split,
                chunk_index,
                rgbs,
                masks,
                poses,
                pos_targets,
                ori_idx,
                shape_ids,
                sample_ids,
                seeds,
                camera_variants,
                augmentation_json,
                shape_families,
                symmetry_orders,
                episode_ids,
                frame_ids,
                depths if depth_available else None,
                compress=compress,
            )
            chunks.append(meta)
            print(f"[collect_dataset] wrote {meta['path']} sha256={meta['sha256'][:12]}...", flush=True)
            if first_chunk is None:
                first_chunk = out_dir / meta["path"]
            rgbs.clear()
            masks.clear()
            poses.clear()
            pos_targets.clear()
            ori_idx.clear()
            shape_ids.clear()
            sample_ids.clear()
            seeds.clear()
            camera_variants.clear()
            augmentation_json.clear()
            shape_families.clear()
            symmetry_orders.clear()
            episode_ids.clear()
            frame_ids.clear()
            depths.clear()
            chunk_index += 1

    env = PegInHoleAlignmentEnv(seed=seed, env_config=env_config, camera_config=camera_config)
    try:
        print(
            f"[collect_dataset] split={split} shapes={len(shapes)} samples_per_shape={samples_per_shape} "
            f"edge_cases={include_edge_cases} target={total_target} chunk_size={chunk_size} "
            f"compress={compress}",
            flush=True,
        )
        for shape in shapes:
            print(f"[collect_dataset] shape={shape}", flush=True)
            if include_edge_cases:
                for pose_error in _edge_pose_errors():
                    sample_seed = seed + sample_id
                    obs, _info = env.reset(
                        seed=sample_seed, options={"shape": shape, "pose_error": pose_error, "nontrivial": False}
                    )
                    add_sample(obs, shape, sample_seed)
            for _ in range(samples_per_shape):
                sample_seed = seed + sample_id
                obs, _info = env.reset(seed=sample_seed, options={"shape": shape})
                add_sample(obs, shape, sample_seed)
    finally:
        env.close()

    if rgbs:
        print(f"[collect_dataset] writing final chunk {chunk_index:03d} ({len(rgbs)} samples)...", flush=True)
        meta = _flush_chunk(
            out_dir,
            split,
            chunk_index,
            rgbs,
            masks,
            poses,
            pos_targets,
            ori_idx,
            shape_ids,
            sample_ids,
            seeds,
            camera_variants,
            augmentation_json,
            shape_families,
            symmetry_orders,
            episode_ids,
            frame_ids,
            depths if depth_available else None,
            compress=compress,
        )
        chunks.append(meta)
        print(f"[collect_dataset] wrote {meta['path']} sha256={meta['sha256'][:12]}...", flush=True)
        if first_chunk is None:
            first_chunk = out_dir / meta["path"]

    class_counts_dict = {str(i): int(class_counts[i]) for i in [0, 1, 2]}
    pose_hist = {
        "dx_mm_min": float(pose_min[0] * 1000) if sample_id else 0.0,
        "dx_mm_max": float(pose_max[0] * 1000) if sample_id else 0.0,
        "dy_mm_min": float(pose_min[1] * 1000) if sample_id else 0.0,
        "dy_mm_max": float(pose_max[1] * 1000) if sample_id else 0.0,
        "dyaw_deg_min": float(pose_min[2]) if sample_id else 0.0,
        "dyaw_deg_max": float(pose_max[2]) if sample_id else 0.0,
    }
    resolved_config = {
        "environment": asdict(env.config),
        "camera": asdict(env.camera_config),
        "collection": {
            "split": split,
            "samples_per_shape": int(samples_per_shape),
            "seed": int(seed),
            "include_edge_cases": bool(include_edge_cases),
            "randomization_level": randomization_level,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_revision": SCHEMA_REVISION,
        "schema_id": SCHEMA_ID,
        "metadata_path": "manifest.json",
        "source_revision": source_revision(),
        "config_hash": canonical_config_hash(resolved_config),
        "resolved_config": resolved_config,
        "randomization": {
            "implementation": "sfn.data.augment",
            "record_version": 1,
            "level": randomization_level,
            "per_sample_field": "augmentation_json",
        },
        "creation_command": " ".join(sys.argv),
        "date_unix": time.time(),
        "split": split,
        "split_definition_version": SPLIT_DEFINITION_VERSION,
        "samples": sample_id,
        "chunks": chunks,
        "seed": seed,
        "shapes": shapes,
        "shape_catalog": shape_catalog(shapes),
        "modalities": {"rgb": {"required": True}, "mask": {"required": True}, "depth": {"required": False, "present": bool(depth_available), "units": "m"}},
        "class_pixel_counts": class_counts_dict,
        "pose_histograms": pose_hist,
        "chunk_size": chunk_size,
        "include_edge_cases": bool(include_edge_cases),
        "compressed": bool(compress),
        "renderer_backend": env.camera_config.renderer_backend,
        "camera_config": camera_contract(env.camera_config),
        "physics_parameters": {
            "engine": "none",
            "dynamics_enabled": False,
            "observation_model": env.camera_config.renderer_backend,
        },
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if first_chunk is None:
        raise ValueError("No samples were collected; use samples_per_shape > 0 or --include-edge-cases")
    report(force=True)
    return first_chunk
