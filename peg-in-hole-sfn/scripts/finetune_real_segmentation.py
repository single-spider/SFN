#!/usr/bin/env python
"""Fine-tune the existing segmentation trainer on a small reviewed real dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.sim2real.annotations import load_png_mask  # noqa: E402
from sfn.training.common import load_checkpoint_cpu, save_checkpoint  # noqa: E402
from sfn.training.perception_cli import add_perception_args, print_result, run_perception_cli  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_real_segmentation_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    *,
    review_manifest: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert matching image/indexed-mask files to the existing NPZ schema."""

    image_dir, mask_dir, output_dir = Path(image_dir), Path(mask_dir), Path(output_dir)
    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    masks = {path.stem: path for path in mask_dir.glob("*.png")}
    allowed: set[str] | None = None
    if review_manifest is not None:
        review = json.loads(Path(review_manifest).read_text(encoding="utf-8"))
        accepted = {"accept", "accepted", "approve", "approved", "corrected"}
        allowed = {
            Path(str(item.get("image", item.get("id", "")))).stem
            for item in review.get("items", [])
            if str(item.get("decision") or item.get("status", "")).lower() in accepted
        }
    stems = sorted(images.keys() & masks.keys())
    if allowed is not None:
        stems = [stem for stem in stems if stem in allowed]
    if not stems:
        raise ValueError("no matching approved image/mask pairs were found")
    manifest_path = output_dir / "manifest.json"
    chunk_path = output_dir / "real_train_000.npz"
    if (manifest_path.exists() or chunk_path.exists()) and not overwrite:
        raise FileExistsError(f"prepared dataset already exists at {output_dir}; pass overwrite=True to replace it")

    rgbs: list[np.ndarray] = []
    semantic_masks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for stem in stems:
        rgb = np.asarray(Image.open(images[stem]).convert("RGB"), dtype=np.uint8)
        mask = load_png_mask(masks[stem])
        if rgb.shape[:2] != mask.shape:
            raise ValueError(f"image/mask dimensions differ for {stem}: {rgb.shape[:2]} != {mask.shape}")
        if expected_shape is None:
            expected_shape = mask.shape
        if mask.shape != expected_shape:
            raise ValueError("all real fine-tuning samples must have the same dimensions")
        if np.any(mask > 2):
            raise ValueError("the existing segmentation model supports semantic class ids 0, 1, and 2")
        rgbs.append(np.transpose(rgb, (2, 0, 1)))
        semantic_masks.append(mask.astype(np.uint8))

    rgb_array = np.stack(rgbs)
    mask_array = np.stack(semantic_masks)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chunk_path,
        rgb=rgb_array,
        mask=mask_array,
        shape_id=np.asarray(["real"] * len(stems)),
        sample_id=np.arange(len(stems), dtype=np.int64),
        source_file=np.asarray([images[stem].name for stem in stems]),
    )
    counts = np.bincount(mask_array.reshape(-1), minlength=3)[:3]
    manifest = {
        "schema_version": 2,
        "metadata_path": "manifest.json",
        "split": "real_finetune",
        "samples": len(stems),
        "chunks": [{"path": chunk_path.name, "sha256": _sha256(chunk_path), "samples": len(stems)}],
        "seed": 0,
        "shapes": ["real"],
        "class_pixel_counts": {str(index): int(value) for index, value in enumerate(counts)},
        "source_images": str(image_dir),
        "source_masks": str(mask_dir),
        "review_manifest": str(review_manifest) if review_manifest is not None else None,
        "date_unix": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir


def make_finetune_initialization(checkpoint: str | Path, output: str | Path) -> Path:
    """Reset trainer state while retaining pretrained model weights."""

    source = load_checkpoint_cpu(checkpoint)
    if "model_state_dict" not in source:
        raise ValueError("initial checkpoint is missing model_state_dict")
    initialized = dict(source)
    initialized.update(
        {
            "epoch": 0,
            "global_step": 0,
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "metrics": {},
            "fine_tune_source": {"path": str(checkpoint), "sha256": _sha256(Path(checkpoint))},
        }
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, initialized)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_perception_args(parser, root=ROOT, task="segmentation", default_config="segmentation.yaml")
    parser.add_argument("--images", type=Path, help="Real RGB image directory; prepares --dataset automatically")
    parser.add_argument("--masks", type=Path, help="Reviewed indexed PNG mask directory")
    parser.add_argument("--prepared-dataset", type=Path, default=ROOT / "data" / "real_finetune")
    parser.add_argument("--review-manifest", type=Path, help="Use only accepted/approved review items")
    parser.add_argument("--overwrite-prepared", action="store_true")
    parser.add_argument(
        "--init-checkpoint", type=Path, help="Pretrained checkpoint; weights are loaded with fresh optimizer"
    )
    args = parser.parse_args()

    if (args.images is None) != (args.masks is None):
        parser.error("--images and --masks must be supplied together")
    if args.images is not None:
        args.dataset = str(
            prepare_real_segmentation_dataset(
                args.images,
                args.masks,
                args.prepared_dataset,
                review_manifest=args.review_manifest,
                overwrite=args.overwrite_prepared,
            )
        )
    if args.init_checkpoint and args.resume:
        parser.error("use either --init-checkpoint for fine-tuning or --resume for an interrupted fine-tune, not both")
    if args.init_checkpoint:
        pretrained = load_checkpoint_cpu(args.init_checkpoint)
        args.base_channels = int(pretrained.get("model_config", {}).get("base", args.base_channels))
        initializer = Path(args.out).with_suffix(".finetune-init.pt")
        args.resume = str(make_finetune_initialization(args.init_checkpoint, initializer))
    elif not args.resume:
        parser.error("fine-tuning requires --init-checkpoint (or --resume for an interrupted run)")
    print_result(run_perception_cli("segmentation", args))


if __name__ == "__main__":
    main()
