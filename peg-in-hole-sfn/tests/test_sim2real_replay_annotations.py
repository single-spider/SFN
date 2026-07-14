from __future__ import annotations

import json

import cv2
import numpy as np
from sfn.sim2real.annotations import export_coco_masks, import_coco_masks, load_png_mask, save_png_mask
from sfn.sim2real.replay import iter_image_folder


def test_image_folder_replay_uses_natural_order_and_rgb(tmp_path):
    for name, red in (("frame10.png", 30), ("frame2.png", 20), ("frame1.png", 10)):
        bgr = np.zeros((3, 4, 3), dtype=np.uint8)
        bgr[..., 2] = red
        assert cv2.imwrite(str(tmp_path / name), bgr)
    frames = list(iter_image_folder(tmp_path, fps=2))
    assert [frame.source.rsplit("frame", 1)[-1] for frame in frames] == ["1.png", "2.png", "10.png"]
    assert [frame.timestamp_s for frame in frames] == [0, 0.5, 1]
    assert int(frames[0].image_rgb[0, 0, 0]) == 10


def test_png_and_coco_polygon_round_trip(tmp_path):
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[3:9, 4:11] = 1
    png = save_png_mask(mask, tmp_path / "masks" / "sample.png")
    np.testing.assert_array_equal(load_png_mask(png), mask)
    coco = export_coco_masks([("sample.png", mask)], tmp_path / "annotations.json")
    payload = json.loads(coco.read_text(encoding="utf-8"))
    assert payload["annotations"][0]["area"] == int(mask.sum())
    restored = import_coco_masks(coco)["sample.png"]
    np.testing.assert_array_equal(restored > 0, mask > 0)


def test_empty_mask_exports_without_annotation(tmp_path):
    output = export_coco_masks([("empty.png", np.zeros((2, 3), dtype=np.uint8))], tmp_path / "empty.json")
    assert json.loads(output.read_text(encoding="utf-8"))["annotations"] == []
