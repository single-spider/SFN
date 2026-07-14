import pytest

torch = pytest.importorskip("torch")

from sfn.data.collect import collect_npz
from sfn.models import VirtualSensorNetwork
from sfn.training.common import load_checkpoint_cpu
from sfn.training.perception import hyperparameter_search, train_perception


def test_segmentation_training_checkpoint_and_vsn_load(tmp_path):
    collect_npz(tmp_path / "data", split="train_seen", samples_per_shape=1, seed=3)
    ckpt_path = tmp_path / "seg.pt"
    result = train_perception(
        "segmentation", tmp_path / "data", ckpt_path, epochs=1, batch_size=2, limit=2, base_channels=4
    )
    assert result["checkpoint"] == str(ckpt_path)
    ckpt = load_checkpoint_cpu(ckpt_path)
    assert ckpt["schema_version"] == 1
    vsn = VirtualSensorNetwork.from_checkpoints(segmentation_path=ckpt_path)
    assert not vsn.training


def test_resumable_training_writes_last_metrics_and_summary(tmp_path):
    collect_npz(tmp_path / "data", split="train_seen", samples_per_shape=1, seed=4, chunk_size=2)
    ckpt_path = tmp_path / "seg.pt"
    result1 = train_perception(
        "segmentation",
        tmp_path / "data",
        ckpt_path,
        epochs=1,
        batch_size=2,
        limit=2,
        val_fraction=0.5,
        progress=False,
    )
    last = tmp_path / "seg.last.pt"
    assert last.exists()
    assert (tmp_path / "seg.metrics.jsonl").exists()
    assert (tmp_path / "seg.summary.json").exists()
    before_resume = {k: v.detach().clone() for k, v in load_checkpoint_cpu(last)["model_state_dict"].items()}
    result2 = train_perception(
        "segmentation",
        tmp_path / "data",
        ckpt_path,
        epochs=2,
        batch_size=2,
        limit=2,
        val_fraction=0.5,
        resume=last,
        progress=False,
    )
    assert result2["epoch"] == 2
    assert result2["global_step"] >= result1["global_step"]
    after_resume = load_checkpoint_cpu(last)["model_state_dict"]
    assert any(not torch.equal(before_resume[k], after_resume[k]) for k in before_resume)


def test_segmentation_without_validation_selects_loss_min(tmp_path):
    collect_npz(tmp_path / "data", split="train_seen", samples_per_shape=1, seed=5)
    result = train_perception(
        "segmentation",
        tmp_path / "data",
        tmp_path / "seg.pt",
        epochs=1,
        batch_size=2,
        limit=2,
        progress=False,
    )
    assert result["validation_source"] == "train"
    assert result["selection_metric"] == "loss"
    assert result["selection_mode"] == "min"


def test_hyperparameter_search_records_actual_selection_metric(tmp_path):
    collect_npz(tmp_path / "data", split="train_seen", samples_per_shape=1, seed=6)
    summary = hyperparameter_search(
        "segmentation",
        tmp_path / "data",
        tmp_path / "search",
        "lr=0.001",
        epochs=1,
        batch_size=2,
        limit=2,
        metric="loss",
        progress=False,
    )
    assert summary["metric"] == "loss"
    assert summary["mode"] == "min"
    assert summary["best"]["checkpoint"]
