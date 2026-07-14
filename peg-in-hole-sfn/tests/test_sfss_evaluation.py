import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sfn.evaluation.evaluate_contract import episode_schedule
from sfn.evaluation.evaluate_oracle import evaluate_oracle
from sfn.evaluation.evaluate_sfss import evaluate_sfss, summarize_episodes
from sfn.models.controllers import SFSSController


class DummyVSNOutput:
    def __init__(self):
        self.dxy_m = torch.tensor([[0.004, -0.002]], dtype=torch.float32)
        self.dyaw_deg = torch.tensor([6.0], dtype=torch.float32)
        self.position_confidence = torch.tensor([1.0], dtype=torch.float32)
        self.orientation_confidence = torch.tensor([1.0], dtype=torch.float32)


def test_sfss_controller_action_signs():
    ctrl = SFSSController(gain_xy=1.0, gain_yaw=1.0, confidence_mode="ignore")
    action = ctrl.act(DummyVSNOutput())
    assert np.allclose(action.physical, [-0.004, 0.002, -6.0])
    assert np.all(action.normalized <= 1.0)
    assert np.all(action.normalized >= -1.0)


def test_sfss_deadband_and_cumulative_motion_hold():
    output = DummyVSNOutput()
    output.dxy_m = torch.tensor([[0.0002, -0.0001]], dtype=torch.float32)
    output.dyaw_deg = torch.tensor([0.1], dtype=torch.float32)
    deadband = SFSSController(
        gain_xy=1.0,
        gain_yaw=1.0,
        confidence_mode="ignore",
        deadband_xy_mm=0.5,
        deadband_yaw_deg=0.2,
    )
    assert np.allclose(deadband.act(output).physical, 0.0)

    limited = SFSSController(
        gain_xy=1.0,
        gain_yaw=1.0,
        confidence_mode="ignore",
        max_cumulative_xy_mm=5.0,
    )
    assert not np.allclose(limited.act(DummyVSNOutput()).physical, 0.0)
    assert np.allclose(limited.act(DummyVSNOutput()).physical, 0.0)
    assert limited.last_hold_reason == "cumulative_motion_limit"


def test_sfss_oscillation_hold_and_reset():
    ctrl = SFSSController(gain_xy=1.0, gain_yaw=1.0, confidence_mode="ignore", max_sign_reversals=1)
    output = DummyVSNOutput()
    ctrl.act(output)
    output.dxy_m *= -1
    output.dyaw_deg *= -1
    ctrl.act(output)
    output.dxy_m *= -1
    output.dyaw_deg *= -1
    assert np.allclose(ctrl.act(output).physical, 0.0)
    assert ctrl.last_hold_reason == "oscillation"
    ctrl.reset()
    assert ctrl.sign_reversals == 0


def test_sfss_evaluator_runs_one_step_with_default_vsn():
    records, steps = evaluate_sfss(
        shapes=["synthetic-square"],
        episodes_per_shape=1,
        seed=21,
        mask_source="ground_truth",
        recursive=False,
        confidence_mode="ignore",
    )
    assert len(records) == 1
    assert len(steps) == 1
    summary = summarize_episodes(records)
    assert summary["episodes"] == 1
    assert "success_rate" in summary
    assert "success_rate_wilson_95" in summary
    assert records[0]["renderer_backend"]
    assert records[0]["episode_seed"] == 21
    assert steps[0]["method"] == "sfss_one_step"


def test_total_episode_budget_is_balanced_and_reproducible():
    schedule = episode_schedule(["a", "b"], episodes=5, seed=17)
    assert [spec.shape for spec in schedule] == ["a", "b", "a", "b", "a"]
    assert [spec.shape_episode for spec in schedule] == [0, 0, 1, 1, 2]
    assert [spec.seed for spec in schedule] == [17, 18, 19, 20, 21]

    first, _ = evaluate_sfss(
        shapes=["synthetic-square", "synthetic-round"],
        episodes=3,
        seed=31,
        mask_source="ground_truth",
        recursive=False,
        confidence_mode="ignore",
    )
    second, _ = evaluate_sfss(
        shapes=["synthetic-square", "synthetic-round"],
        episodes=3,
        seed=31,
        mask_source="ground_truth",
        recursive=False,
        confidence_mode="ignore",
    )
    assert len(first) == 3
    assert [row["shape"] for row in first] == ["synthetic-square", "synthetic-round", "synthetic-square"]
    state_fields = ("episode_id", "episode_seed", "initial_dx_m", "initial_dy_m", "initial_dyaw_deg")
    assert [[row[field] for field in state_fields] for row in first] == [
        [row[field] for field in state_fields] for row in second
    ]
    oracle, _ = evaluate_oracle(
        shapes=["synthetic-square", "synthetic-round"],
        episodes=3,
        seed=31,
        task="alignment",
    )
    assert [[row[field] for field in state_fields] for row in first] == [
        [row[field] for field in state_fields] for row in oracle
    ]


def test_total_and_legacy_episode_arguments_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        evaluate_sfss(
            shapes=["synthetic-square"],
            episodes=1,
            episodes_per_shape=1,
            recursive=False,
        )
