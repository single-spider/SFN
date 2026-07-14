"""Convert indexed PNG masks to/from COCO, CVAT, or Label Studio annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.sim2real.annotations import (
    export_coco_masks,
    export_cvat_masks,
    export_label_studio_masks,
    import_coco_masks,
    import_cvat_masks,
    import_label_studio_masks,
    load_png_mask,
    save_png_mask,
)


def _categories(value: Path | None):
    if value is None:
        return None
    import json

    payload = json.loads(value.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return {int(key): str(name) for key, name in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("export-coco", "export-cvat", "export-label-studio"):
        export = sub.add_parser(command)
        export.add_argument("mask_dir", type=Path)
        export.add_argument("output_annotations", type=Path)
        export.add_argument("--categories", type=Path, help="JSON id-to-name mapping or indexed name list")
        if command == "export-coco":
            export.add_argument("--category", default="foreground", help="Legacy name for semantic class 1")
        if command == "export-label-studio":
            export.add_argument("--image-prefix", default="")
    for command in ("import-coco", "import-cvat", "import-label-studio"):
        restore = sub.add_parser(command)
        restore.add_argument("input_annotations", type=Path)
        restore.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    if args.command.startswith("export-"):
        paths = sorted(args.mask_dir.glob("*.png"))
        if not paths:
            parser.error(f"no PNG masks found in {args.mask_dir}")
        records = [(path.name, load_png_mask(path)) for path in paths]
        categories = _categories(args.categories)
        if args.command == "export-coco":
            export_coco_masks(records, args.output_annotations, category_name=args.category, categories=categories)
        elif args.command == "export-cvat":
            export_cvat_masks(records, args.output_annotations, categories=categories)
        else:
            export_label_studio_masks(
                records, args.output_annotations, categories=categories, image_prefix=args.image_prefix
            )
    else:
        importer = {
            "import-coco": import_coco_masks,
            "import-cvat": import_cvat_masks,
            "import-label-studio": import_label_studio_masks,
        }[args.command]
        for file_name, mask in importer(args.input_annotations).items():
            save_png_mask(mask, args.output_dir / (Path(file_name).stem + ".png"))


if __name__ == "__main__":
    main()
