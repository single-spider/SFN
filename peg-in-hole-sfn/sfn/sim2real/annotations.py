"""Interoperability helpers for semantic PNG masks and annotation tools.

The on-disk source of truth is an indexed PNG: pixel value ``0`` is normally
background and every non-zero value is a semantic category id.  COCO, CVAT and
Label Studio conversions deliberately retain those ids rather than collapsing
all foreground pixels into one class.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

CategorySpec = Mapping[int | str, str] | Sequence[str] | None


def load_png_mask(path: str | Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D indexed PNG")
    return mask.astype(np.int32, copy=False)


def save_png_mask(mask: np.ndarray, path: str | Path) -> Path:
    array = np.asarray(mask)
    if array.ndim != 2 or np.any(array < 0) or np.any(array > 65535):
        raise ValueError("mask must be 2-D with labels in [0, 65535]")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8 if array.max(initial=0) <= 255 else np.uint16)).save(output)
    return output


def _category_map(categories: CategorySpec, labels: Iterable[int], category_name: str = "foreground") -> dict[int, str]:
    label_ids = sorted({int(value) for value in labels if int(value) != 0})
    if categories is None:
        return {value: category_name if value == 1 else f"class_{value}" for value in label_ids}
    if isinstance(categories, Mapping):
        result = {int(key): str(value) for key, value in categories.items() if int(key) != 0}
    else:
        # A sequence is interpreted as an indexed class-name table, including
        # background at index zero.  This mirrors common segmentation configs.
        result = {index: str(name) for index, name in enumerate(categories) if index != 0}
    for value in label_ids:
        result.setdefault(value, category_name if value == 1 else f"class_{value}")
    return dict(sorted(result.items()))


def _mask_polygons(binary: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [
        contour.reshape(-1, 2).astype(float).ravel().tolist()
        for contour in contours
        if contour.reshape(-1, 2).shape[0] >= 3
    ]


def _records(records: Iterable[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    materialized: list[tuple[str, np.ndarray]] = []
    for file_name, raw_mask in records:
        mask = np.asarray(raw_mask)
        if mask.ndim != 2:
            raise ValueError("each mask must be 2-D")
        if np.any(mask < 0) or np.any(mask > 65535):
            raise ValueError("semantic labels must be in [0, 65535]")
        materialized.append((str(file_name), mask.astype(np.uint16, copy=False)))
    return materialized


def export_coco_masks(
    records: Iterable[tuple[str, np.ndarray]],
    path: str | Path,
    *,
    category_name: str = "foreground",
    categories: CategorySpec = None,
) -> Path:
    """Export indexed masks as polygon COCO annotations.

    One annotation is emitted per image/category pair.  Category ids equal the
    PNG pixel values, which makes a round trip lossless with respect to semantic
    classes (apart from the usual polygon boundary approximation).
    """

    items = _records(records)
    category_map = _category_map(categories, (v for _, mask in items for v in np.unique(mask)), category_name)
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for image_id, (file_name, mask) in enumerate(items, 1):
        height, width = mask.shape
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
        for category_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
            binary = (mask == category_id).astype(np.uint8)
            polygons = _mask_polygons(binary)
            if not polygons:
                continue
            x, y, w, h = cv2.boundingRect(binary)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": polygons,
                    "area": int(binary.sum()),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": key, "name": value} for key, value in category_map.items()],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def _decode_uncompressed_rle(segmentation: Mapping[str, Any], height: int, width: int) -> np.ndarray:
    counts = segmentation.get("counts")
    if not isinstance(counts, list):
        raise NotImplementedError("compressed COCO RLE requires optional pycocotools")
    values: list[int] = []
    bit = 0
    for run in counts:
        run_int = int(run)
        if run_int < 0:
            raise ValueError("COCO RLE counts cannot be negative")
        values.extend([bit] * run_int)
        bit = 1 - bit
    if len(values) != height * width:
        raise ValueError("COCO RLE length does not match image dimensions")
    return np.asarray(values, dtype=np.uint8).reshape((height, width), order="F")


def import_coco_masks(path: str | Path) -> dict[str, np.ndarray]:
    """Rasterize polygon or uncompressed-RLE COCO annotations.

    Semantic pixel values are the original ``category_id`` values.  Overlapping
    annotations are applied in annotation order, matching common COCO tooling.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload.get("images", [])}
    masks = {
        str(item["file_name"]): np.zeros((int(item["height"]), int(item["width"])), dtype=np.uint16)
        for item in images.values()
    }
    for annotation in payload.get("annotations", []):
        image = images.get(int(annotation["image_id"]))
        if image is None:
            raise ValueError(f"annotation references unknown image_id {annotation['image_id']}")
        category_id = int(annotation.get("category_id", 1))
        if not 0 <= category_id <= 65535:
            raise ValueError(f"category_id {category_id} cannot be represented by an indexed PNG")
        target = masks[str(image["file_name"])]
        segmentation = annotation.get("segmentation", [])
        if isinstance(segmentation, Mapping):
            binary = _decode_uncompressed_rle(segmentation, target.shape[0], target.shape[1])
            target[binary.astype(bool)] = category_id
            continue
        polygons = [
            np.asarray(points, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            for points in segmentation
            if len(points) >= 6
        ]
        if polygons:
            cv2.fillPoly(target, polygons, category_id)
    return masks


def export_cvat_masks(
    records: Iterable[tuple[str, np.ndarray]], path: str | Path, *, categories: CategorySpec = None
) -> Path:
    """Export CVAT's native image-annotation XML with semantic polygons."""

    items = _records(records)
    category_map = _category_map(categories, (v for _, mask in items for v in np.unique(mask)))
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    labels = ET.SubElement(task, "labels")
    for category_id, name in category_map.items():
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "attributes")
        ET.SubElement(label, "id").text = str(category_id)
    for image_id, (file_name, mask) in enumerate(items):
        image = ET.SubElement(
            root, "image", id=str(image_id), name=file_name, width=str(mask.shape[1]), height=str(mask.shape[0])
        )
        for category_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
            for polygon in _mask_polygons(mask == category_id):
                points = ";".join(f"{polygon[i]:g},{polygon[i + 1]:g}" for i in range(0, len(polygon), 2))
                ET.SubElement(
                    image,
                    "polygon",
                    label=category_map[category_id],
                    points=points,
                    occluded="0",
                    source="manual",
                    z_order="0",
                    **{"class_id": str(category_id)},
                )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output


def import_cvat_masks(path: str | Path) -> dict[str, np.ndarray]:
    """Import CVAT image-annotation XML exported by this module or CVAT."""

    root = ET.parse(path).getroot()
    labels = [node.text or "" for node in root.findall("./meta/task/labels/label/name")]
    label_ids: dict[str, int] = {}
    for index, label in enumerate(root.findall("./meta/task/labels/label"), 1):
        name = label.findtext("name", f"class_{index}")
        label_ids[name] = int(label.findtext("id", str(index)))
    if not label_ids:
        label_ids = {name: index for index, name in enumerate(labels, 1)}
    masks: dict[str, np.ndarray] = {}
    for image in root.findall("image"):
        name = image.attrib["name"]
        mask = np.zeros((int(image.attrib["height"]), int(image.attrib["width"])), dtype=np.uint16)
        for polygon in image.findall("polygon"):
            label = polygon.attrib.get("label", "foreground")
            category_id = int(polygon.attrib.get("class_id", label_ids.setdefault(label, len(label_ids) + 1)))
            points = (
                np.asarray(
                    [
                        [float(value) for value in pair.split(",")]
                        for pair in polygon.attrib.get("points", "").split(";")
                        if pair
                    ],
                    dtype=np.float32,
                )
                .round()
                .astype(np.int32)
            )
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], category_id)
        masks[name] = mask
    return masks


def export_label_studio_masks(
    records: Iterable[tuple[str, np.ndarray]],
    path: str | Path,
    *,
    categories: CategorySpec = None,
    image_prefix: str = "",
) -> Path:
    """Export Label Studio JSON tasks using ``polygonlabels`` percentages."""

    items = _records(records)
    category_map = _category_map(categories, (v for _, mask in items for v in np.unique(mask)))
    tasks: list[dict[str, Any]] = []
    result_id = 1
    for file_name, mask in items:
        height, width = mask.shape
        results: list[dict[str, Any]] = []
        for category_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
            for polygon in _mask_polygons(mask == category_id):
                points = [
                    [100.0 * polygon[i] / width, 100.0 * polygon[i + 1] / height] for i in range(0, len(polygon), 2)
                ]
                results.append(
                    {
                        "id": str(result_id),
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygonlabels",
                        "original_width": width,
                        "original_height": height,
                        "image_rotation": 0,
                        "value": {"points": points, "polygonlabels": [category_map[category_id]]},
                        "meta": {"category_id": category_id},
                    }
                )
                result_id += 1
        tasks.append(
            {
                "data": {"image": f"{image_prefix}{file_name}"},
                "annotations": [{"result": results}],
                "meta": {"file_name": file_name, "categories": {str(k): v for k, v in category_map.items()}},
            }
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    return output


def import_label_studio_masks(path: str | Path) -> dict[str, np.ndarray]:
    """Import Label Studio polygon-label task JSON into indexed masks."""

    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(tasks, Mapping):
        tasks = [tasks]
    masks: dict[str, np.ndarray] = {}
    inferred: dict[str, int] = {}
    for task in tasks:
        meta = task.get("meta", {})
        category_names = {str(name): int(key) for key, name in meta.get("categories", {}).items()}
        results = [result for annotation in task.get("annotations", []) for result in annotation.get("result", [])]
        results += [result for prediction in task.get("predictions", []) for result in prediction.get("result", [])]
        polygon_results = [result for result in results if result.get("type") == "polygonlabels"]
        width = next(
            (int(result.get("original_width", 0)) for result in polygon_results if result.get("original_width")), 0
        )
        height = next(
            (int(result.get("original_height", 0)) for result in polygon_results if result.get("original_height")), 0
        )
        if not width or not height:
            width, height = int(meta.get("width", 0)), int(meta.get("height", 0))
        if not width or not height:
            raise ValueError("Label Studio task is missing original image dimensions")
        image_value = str(task.get("data", {}).get("image", "image.png"))
        file_name = str(meta.get("file_name") or Path(image_value.split("?", 1)[0]).name)
        mask = np.zeros((height, width), dtype=np.uint16)
        for result in polygon_results:
            value = result.get("value", {})
            names = value.get("polygonlabels", [])
            if not names:
                continue
            label = str(names[0])
            explicit = result.get("meta", {}).get("category_id")
            if explicit is None:
                explicit = category_names.get(label)
            if explicit is None:
                explicit = inferred.setdefault(label, len(inferred) + 1)
            points = (
                np.asarray(
                    [[float(x) * width / 100.0, float(y) * height / 100.0] for x, y in value.get("points", [])],
                    dtype=np.float32,
                )
                .round()
                .astype(np.int32)
            )
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], int(explicit))
        masks[file_name] = mask
    return masks


# Explicit aliases make the API easy to discover from both mask-centric and
# annotation-centric callers.
export_cvat_annotations = export_cvat_masks
import_cvat_annotations = import_cvat_masks
export_label_studio_annotations = export_label_studio_masks
import_label_studio_annotations = import_label_studio_masks
