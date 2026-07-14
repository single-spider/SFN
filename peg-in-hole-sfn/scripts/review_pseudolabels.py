#!/usr/bin/env python
"""Create an auditable review queue for generated segmentation masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.sim2real.active_learning import write_pseudolabel_review_manifest  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("mask_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selection", type=Path, help="Optional active-learning selection JSON")
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--categories", type=Path, help="Optional JSON category map")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    images = {path.stem: path for path in args.image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
    masks = {path.stem: path for path in args.mask_dir.glob("*.png")}
    selected: dict[str, dict] = {}
    if args.selection:
        payload = json.loads(args.selection.read_text(encoding="utf-8"))
        for item in payload.get("items", payload):
            selected[Path(str(item.get("frame", item.get("image", item.get("id"))))).stem] = item
    stems = sorted((images.keys() & masks.keys()) if not selected else (images.keys() & masks.keys() & selected.keys()))
    if not stems:
        parser.error("no matching image/mask stems were found")
    records = []
    for stem in stems:
        source = selected.get(stem, {})
        records.append(
            {
                "id": stem,
                "image": str(images[stem]),
                "mask": str(masks[stem]),
                "uncertainty": source.get("uncertainty"),
                "selection_rank": source.get("selection_rank"),
                "source_index": source.get("source_index"),
            }
        )
    categories = json.loads(args.categories.read_text(encoding="utf-8")) if args.categories else None
    write_pseudolabel_review_manifest(
        records, args.output, model_checkpoint=args.model_checkpoint, categories=categories, seed=args.seed
    )
    print(json.dumps({"output": str(args.output), "pending": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
