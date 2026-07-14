import math

import pytest

torch = pytest.importorskip("torch")

from sfn.evaluation.evaluate_mfms import evaluate_mfms
from sfn.training.train_mfms import (
    MFMSActorCritic,
    MFMSTeacherPretrainConfig,
    MFMSTrainConfig,
    make_mfms_history_state,
    pretrain_mfms_from_sfss_teacher,
    train_mfms,
)


def test_mfms_history_state_is_padded():
    state = torch.ones(452)
    seq = make_mfms_history_state([state], history_len=4, device="cpu")
    assert seq.shape == (1, 4, 452)
    assert torch.all(seq[0, :3] == 0)
    assert torch.all(seq[0, 3] == 1)


def test_mfms_actor_critic_shapes():
    model = MFMSActorCritic()
    mean, value, _hidden = model(torch.zeros(2, 4, 452), lengths=torch.tensor([1, 4]))
    assert mean.shape == (2, 3)
    assert value.shape == (2,)


def test_mfms_teacher_pretrain_and_evaluate_smoke(tmp_path):
    ckpt = tmp_path / "mfms.pt"
    metrics = pretrain_mfms_from_sfss_teacher(
        ckpt,
        config=MFMSTeacherPretrainConfig(
            samples=8,
            epochs=1,
            batch_size=4,
            history_len=4,
            seed=62,
            device="cpu",
        ),
        shapes=["synthetic-square"],
    )
    assert ckpt.exists()
    assert math.isfinite(metrics["loss"])
    records = evaluate_mfms(ckpt, shapes=["synthetic-square"], episodes_per_shape=1, seed=63)
    assert len(records) == 1
    assert records[0]["method"] == "mfms"


def test_mfms_rl_uses_masks_burn_in_logging_and_periodic_checkpoint(tmp_path):
    ckpt = tmp_path / "mfms.pt"
    log = tmp_path / "mfms.jsonl"
    metrics = train_mfms(
        ckpt,
        config=MFMSTrainConfig(
            updates=1,
            rollout_steps=3,
            burn_in_steps=1,
            checkpoint_every=1,
            log_jsonl=str(log),
            seed=64,
            device="cpu",
            anchor_imitation_coef=0.0,
        ),
        shapes=["synthetic-square"],
    )
    assert math.isfinite(metrics["loss"])
    assert log.read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "mfms.update000001.pt").is_file()
    saved = torch.load(ckpt, map_location="cpu")
    assert saved["compatibility"]["mask_source"] == "ground_truth"
