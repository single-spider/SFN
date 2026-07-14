import pytest
from sfn.data.collect import collect_npz
from sfn.evaluation.visuals import generate_dataset_visuals


def test_dataset_visual_panel_generation(tmp_path):
    pytest.importorskip("PIL")
    collect_npz(tmp_path / "data", split="train_seen", samples_per_shape=1, seed=31)
    paths = generate_dataset_visuals(tmp_path / "data", tmp_path / "viz", count=2)
    assert len(paths) == 2
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_segmentation_evaluation_metrics(tmp_path):
    pytest.importorskip("torch")
    from sfn.evaluation.evaluate_perception import evaluate_segmentation
    from sfn.training.perception import train_perception

    collect_npz(tmp_path / "data", split="train_seen", samples_per_shape=1, seed=32)
    ckpt = tmp_path / "seg.pt"
    train_perception("segmentation", tmp_path / "data", ckpt, epochs=1, batch_size=2, limit=2)
    metrics = evaluate_segmentation(tmp_path / "data", ckpt, limit=2)
    assert metrics["samples"] == 2
    assert 0.0 <= metrics["pixel_accuracy"] <= 1.0
    assert 0.0 <= metrics["mean_iou"] <= 1.0
