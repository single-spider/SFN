import argparse
import json
import os
from pathlib import Path

import gym
import gymEnv  # noqa: F401 - registers the local gym environment
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
COMPLEX_DIR = SCRIPT_DIR / "gymEnv" / "envs" / "complex"


def discover_peg_types():
    peg_types = []
    for path in sorted(COMPLEX_DIR.iterdir()):
        if not path.is_dir():
            continue
        if (path / "base" / "base.urdf").exists() and (path / "peg" / "peg.urdf").exists():
            peg_types.append(path.name)
    return peg_types


def build_position_gt(dxy_m):
    dx = np.clip(np.around(-dxy_m[:, 0] * 1000.0) + 10, 0, 20).astype(np.int64)
    dy = np.clip(np.around(dxy_m[:, 1] * 1000.0) + 10, 0, 20).astype(np.int64)

    position_gt = np.zeros((len(dxy_m), 21, 21), dtype=np.float32)
    position_gt[np.arange(len(dxy_m)), dy, dx] = 1.0
    return position_gt


def obs_to_sample(obs):
    img = np.asarray(obs["img"], dtype=np.uint8)
    mask = np.asarray(obs["gt"], dtype=np.uint8)
    dxy_m = np.asarray(obs["dxy"], dtype=np.float32)
    dyaw_deg = np.float32(obs["dyaw"])
    return img, mask, dxy_m, dyaw_deg


def save_preview(preview_dir, peg_type, sample_index, img, mask):
    preview_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.transpose(img, (1, 2, 0))
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask == 1] = [0, 220, 60]
    mask_rgb[mask == 2] = [255, 120, 20]

    overlay = np.clip(0.65 * rgb + 0.35 * mask_rgb, 0, 255).astype(np.uint8)
    stem = f"{peg_type}_{sample_index:05d}"
    Image.fromarray(rgb).save(preview_dir / f"{stem}_rgb.png")
    Image.fromarray(mask_rgb).save(preview_dir / f"{stem}_mask.png")
    Image.fromarray(overlay).save(preview_dir / f"{stem}_overlay.png")


def flush_chunk(output_dir, peg_type, chunk_index, chunk, overwrite):
    if not chunk["img"]:
        return None

    target = output_dir / f"{peg_type}_{chunk_index:04d}.npz"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Pass --overwrite to replace it.")

    imgs = np.stack(chunk["img"]).astype(np.uint8)
    masks = np.stack(chunk["mask"]).astype(np.uint8)
    dxy_m = np.stack(chunk["dxy_m"]).astype(np.float32)
    dyaw_deg = np.asarray(chunk["dyaw_deg"], dtype=np.float32)
    sample_index = np.asarray(chunk["sample_index"], dtype=np.int32)

    np.savez_compressed(
        target,
        rgb=imgs,
        mask=masks,
        dxy_m=dxy_m,
        dyaw_deg=dyaw_deg,
        position_gt=build_position_gt(dxy_m),
        peg_type=np.asarray(peg_type),
        sample_index=sample_index,
    )

    return {
        "path": target.name,
        "peg_type": peg_type,
        "samples": int(len(sample_index)),
        "first_sample_index": int(sample_index[0]),
        "last_sample_index": int(sample_index[-1]),
    }


def collect_for_shape(args, output_dir, peg_type, shape_idx):
    env = gym.make(
        "gymEnv:peg-in-hole-v11",
        peg_type=peg_type,
        seed=args.seed + shape_idx,
        test_mode=False,
        disable_env_checker=True,
    )

    chunk = {"img": [], "mask": [], "dxy_m": [], "dyaw_deg": [], "sample_index": []}
    files = []
    chunk_index = 0

    try:
        env.reset()
        for sample_index in range(args.samples_per_shape):
            obs, _, _, _ = env.step([0.0, 0.0, 0.0])
            img, mask, dxy_m, dyaw_deg = obs_to_sample(obs)

            if sample_index < args.preview_count:
                save_preview(output_dir / "previews", peg_type, sample_index, img, mask)

            chunk["img"].append(img)
            chunk["mask"].append(mask)
            chunk["dxy_m"].append(dxy_m)
            chunk["dyaw_deg"].append(dyaw_deg)
            chunk["sample_index"].append(sample_index)

            if len(chunk["img"]) >= args.chunk_size:
                files.append(flush_chunk(output_dir, peg_type, chunk_index, chunk, args.overwrite))
                chunk = {"img": [], "mask": [], "dxy_m": [], "dyaw_deg": [], "sample_index": []}
                chunk_index += 1

        chunk_file = flush_chunk(output_dir, peg_type, chunk_index, chunk, args.overwrite)
        if chunk_file is not None:
            files.append(chunk_file)
    finally:
        env.close()

    return files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect simulated RGB/mask/alignment samples from peg-in-hole-v11."
    )
    parser.add_argument(
        "--peg_types",
        nargs="+",
        default=None,
        help="Peg types to collect. Defaults to every valid asset under gymEnv/envs/complex.",
    )
    parser.add_argument("--samples_per_shape", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=Path("data/sim_samples"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--preview_count", type=int, default=5)
    parser.add_argument(
        "--opengl_platform",
        default=None,
        help="Optional PYOPENGL_PLATFORM value, for example egl or osmesa.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_shape <= 0:
        raise ValueError("--samples_per_shape must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.preview_count < 0:
        raise ValueError("--preview_count must be non-negative")

    os.chdir(SCRIPT_DIR)
    if args.opengl_platform:
        os.environ["PYOPENGL_PLATFORM"] = args.opengl_platform

    available = discover_peg_types()
    peg_types = args.peg_types or available
    unknown = sorted(set(peg_types) - set(available))
    if unknown:
        raise ValueError(f"Unknown peg type(s): {', '.join(unknown)}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metadata_path} already exists. Pass --overwrite to replace it.")

    metadata = {
        "env_id": "gymEnv:peg-in-hole-v11",
        "seed": args.seed,
        "samples_per_shape": args.samples_per_shape,
        "chunk_size": args.chunk_size,
        "peg_types": peg_types,
        "schema": {
            "rgb": "uint8 array shaped [N, 3, 200, 250]",
            "mask": "uint8 array shaped [N, 200, 250], classes 0=background, 1=peg, 2=seam/hole",
            "dxy_m": "float32 array shaped [N, 2], XY alignment error in meters",
            "dyaw_deg": "float32 array shaped [N], yaw error in degrees",
            "position_gt": "float32 one-hot array shaped [N, 21, 21]",
        },
        "files": [],
    }

    for shape_idx, peg_type in enumerate(peg_types):
        print(f"collecting {args.samples_per_shape} samples for {peg_type}")
        metadata["files"].extend(collect_for_shape(args, output_dir, peg_type, shape_idx))

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    total = sum(item["samples"] for item in metadata["files"])
    print(f"saved {total} samples to {output_dir}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
