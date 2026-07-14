from __future__ import annotations

import json

import numpy as np
from sfn.sim2real.active_learning import (
    build_pseudolabel_review_manifest,
    entropy_uncertainty,
    select_active_learning_indices,
    write_pseudolabel_review_manifest,
)
from sfn.sim2real.annotations import (
    export_coco_masks,
    export_cvat_masks,
    export_label_studio_masks,
    import_coco_masks,
    import_cvat_masks,
    import_label_studio_masks,
)


def _semantic_mask():
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[2:9, 3:11] = 1
    mask[11:18, 13:22] = 2
    return mask


def test_coco_preserves_semantic_category_ids(tmp_path):
    mask = _semantic_mask()
    output = export_coco_masks([("frame.png", mask)], tmp_path / "annotations.json", categories={1: "peg", 2: "hole"})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["categories"] == [{"id": 1, "name": "peg"}, {"id": 2, "name": "hole"}]
    assert {item["category_id"] for item in payload["annotations"]} == {1, 2}
    np.testing.assert_array_equal(import_coco_masks(output)["frame.png"], mask)


def test_cvat_and_label_studio_semantic_round_trip(tmp_path):
    mask = _semantic_mask()
    records = [("frame.png", mask)]
    categories = {1: "peg", 2: "hole"}
    cvat = export_cvat_masks(records, tmp_path / "cvat.xml", categories=categories)
    studio = export_label_studio_masks(records, tmp_path / "label-studio.json", categories=categories)
    np.testing.assert_array_equal(import_cvat_masks(cvat)["frame.png"], mask)
    np.testing.assert_array_equal(import_label_studio_masks(studio)["frame.png"], mask)


def test_entropy_and_diversity_selection_are_deterministic():
    probabilities = np.asarray(
        [
            [[0.99], [0.01]],
            [[0.50], [0.50]],
            [[0.55], [0.45]],
            [[0.60], [0.40]],
        ]
    )
    uncertainty = entropy_uncertainty(probabilities)
    assert int(np.argmax(uncertainty)) == 1
    features = np.asarray([[0.0], [0.1], [0.2], [10.0]])
    first = select_active_learning_indices(uncertainty, 3, features=features, diversity_weight=0.8, seed=17)
    second = select_active_learning_indices(uncertainty, 3, features=features, diversity_weight=0.8, seed=17)
    assert first == second
    assert first[0] == 1
    assert 3 in first  # diversity includes the distant feature rather than three near-duplicates


def test_pseudolabel_review_manifest_is_auditable(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    records = [
        {"id": "f1", "image": "frames/f1.png", "mask": "masks/f1.png", "uncertainty": 0.7},
        ("frames/f2.png", "masks/f2.png"),
    ]
    manifest = build_pseudolabel_review_manifest(records, model_checkpoint=checkpoint, categories={1: "peg"})
    assert manifest["summary"] == {"total": 2, "pending": 2}
    assert len(manifest["model"]["sha256"]) == 64
    assert all(item["status"] == "pending" and "decision" in item for item in manifest["items"])
    output = write_pseudolabel_review_manifest(records, tmp_path / "review.json")
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
