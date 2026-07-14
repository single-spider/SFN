import math

import pytest

torch = pytest.importorskip("torch")

from sfn.evaluation.evaluate_sfms import evaluate_sfms
from sfn.training.train_mfms import MFMSTrainConfig, recurrent_context_after_boundary, train_mfms
from sfn.training.train_sfms import SFMSTrainConfig, make_sfms_state, random_policy_smoke, train_sfms


def test_sfms_state_is_canonical_452_values():
    class Dummy:
        position_prob = torch.zeros(2, 21, 21)
        orientation_prob = torch.zeros(2, 11)

    state = make_sfms_state(Dummy())
    assert state.shape == (2, 452)


def test_policy_state_cannot_consume_privileged_pose_or_robot_fields():
    class Dummy:
        position_prob = torch.arange(2 * 21 * 21, dtype=torch.float32).reshape(2, 21, 21)
        orientation_prob = torch.arange(2 * 11, dtype=torch.float32).reshape(2, 11)
        pose_error = torch.tensor([999.0, 999.0, 999.0])
        robot_joint_state = torch.ones(7)

    baseline = make_sfms_state(Dummy()).clone()
    Dummy.pose_error = torch.tensor([-999.0, -999.0, -999.0])
    Dummy.robot_joint_state = torch.zeros(7)
    assert torch.equal(baseline, make_sfms_state(Dummy()))
    assert baseline.shape == (2, 452)


def test_random_policy_smoke_has_finite_metrics():
    metrics = random_policy_smoke(episodes=1, shapes=["synthetic-square"], seed=55)
    assert metrics["episodes"] == 1
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert math.isfinite(metrics["mean_reward"])


def test_sfms_train_and_evaluate_smoke(tmp_path):
    ckpt = tmp_path / "sfms.pt"
    metrics = train_sfms(
        ckpt,
        config=SFMSTrainConfig(updates=1, rollout_steps=2, seed=56, device="cpu"),
        shapes=["synthetic-square"],
    )
    assert ckpt.exists()
    assert math.isfinite(metrics["loss"])
    records = evaluate_sfms(ckpt, shapes=["synthetic-square"], episodes_per_shape=1, seed=57)
    assert len(records) == 1
    assert "success" in records[0]


def test_sfms_vector_logging_periodic_checkpoint_and_compatibility(tmp_path):
    ckpt = tmp_path / "sfms.pt"
    log = tmp_path / "metrics.jsonl"
    metrics = train_sfms(
        ckpt,
        config=SFMSTrainConfig(
            updates=1,
            rollout_steps=2,
            num_envs=2,
            checkpoint_every=1,
            log_jsonl=str(log),
            seed=560,
            device="cpu",
        ),
        shapes=["synthetic-square"],
    )
    assert metrics["global_step"] == 4
    assert log.read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "sfms.update000001.pt").is_file()
    saved = torch.load(ckpt, map_location="cpu")
    assert saved["compatibility"]["mask_source"] == "ground_truth"


def test_sfms_resume_is_exact_at_update_boundary(tmp_path):
    uninterrupted = tmp_path / "uninterrupted.pt"
    first = tmp_path / "first.pt"
    resumed = tmp_path / "resumed.pt"
    common = dict(rollout_steps=2, num_envs=2, seed=561, device="cpu")
    train_sfms(
        uninterrupted,
        config=SFMSTrainConfig(updates=2, log_jsonl=str(tmp_path / "u.jsonl"), **common),
        shapes=["synthetic-square"],
    )
    train_sfms(
        first,
        config=SFMSTrainConfig(updates=1, log_jsonl=str(tmp_path / "r.jsonl"), **common),
        shapes=["synthetic-square"],
    )
    train_sfms(
        resumed,
        config=SFMSTrainConfig(updates=1, log_jsonl=str(tmp_path / "r.jsonl"), **common),
        shapes=["synthetic-square"],
        resume_path=first,
    )
    left = torch.load(uninterrupted, map_location="cpu")
    right = torch.load(resumed, map_location="cpu")
    assert left["global_step"] == right["global_step"] == 8
    for name, tensor in left["model_state_dict"].items():
        assert torch.equal(tensor, right["model_state_dict"][name]), name


def test_mfms_resume_is_exact_at_update_boundary(tmp_path):
    uninterrupted = tmp_path / "uninterrupted_mfms.pt"
    first = tmp_path / "first_mfms.pt"
    resumed = tmp_path / "resumed_mfms.pt"
    common = dict(rollout_steps=2, history_len=2, seed=562, device="cpu")
    train_mfms(
        uninterrupted,
        config=MFMSTrainConfig(updates=2, log_jsonl=str(tmp_path / "mu.jsonl"), **common),
        shapes=["synthetic-square"],
    )
    train_mfms(
        first,
        config=MFMSTrainConfig(updates=1, log_jsonl=str(tmp_path / "mr.jsonl"), **common),
        shapes=["synthetic-square"],
    )
    train_mfms(
        resumed,
        config=MFMSTrainConfig(updates=1, log_jsonl=str(tmp_path / "mr.jsonl"), **common),
        shapes=["synthetic-square"],
        resume_path=first,
    )
    left = torch.load(uninterrupted, map_location="cpu")
    right = torch.load(resumed, map_location="cpu")
    assert left["global_step"] == right["global_step"] == 4
    for name, tensor in left["model_state_dict"].items():
        assert torch.equal(tensor, right["model_state_dict"][name]), name


@pytest.mark.parametrize(("terminated", "truncated"), [(True, False), (False, True)])
def test_mfms_recurrent_context_resets_for_both_boundary_types(terminated, truncated):
    assert recurrent_context_after_boundary([torch.ones(452)], terminated, truncated) == []


def test_sfms_stabilized_finetune_saves_best_checkpoint(tmp_path):
    init_ckpt = tmp_path / "sfms_init.pt"
    best_ckpt = tmp_path / "sfms_best.pt"
    final_ckpt = tmp_path / "sfms_final.pt"
    train_sfms(
        init_ckpt,
        config=SFMSTrainConfig(updates=1, rollout_steps=2, seed=58, device="cpu"),
        shapes=["synthetic-square"],
    )
    metrics = train_sfms(
        final_ckpt,
        config=SFMSTrainConfig(
            updates=1,
            rollout_steps=2,
            seed=59,
            device="cpu",
            actor_lr=1e-5,
            critic_lr=1e-4,
            entropy_coef=0.0,
            anchor_imitation_coef=1.0,
            eval_every=1,
            eval_episodes_per_shape=1,
        ),
        shapes=["synthetic-square"],
        initial_policy_path=init_ckpt,
        best_out=best_ckpt,
    )
    assert final_ckpt.exists()
    assert best_ckpt.exists()
    assert "eval" in metrics


def test_sfms_evaluate_insertion_task_smoke(tmp_path):
    ckpt = tmp_path / "sfms.pt"
    train_sfms(
        ckpt,
        config=SFMSTrainConfig(updates=1, rollout_steps=2, seed=60, device="cpu"),
        shapes=["synthetic-square"],
    )
    records = evaluate_sfms(
        ckpt,
        shapes=["synthetic-square"],
        episodes_per_shape=1,
        seed=61,
        task="insertion",
    )
    assert len(records) == 1
    assert records[0]["task"] == "insertion"
    assert "insertion_depth_mm" in records[0]
