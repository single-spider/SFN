import pytest

torch = pytest.importorskip("torch")

from sfn.training.common import assert_checkpoint_compatible, load_checkpoint_cpu, make_checkpoint, save_checkpoint
from sfn.training.rl_utils import compute_gae, last_valid_indices, tanh_normal_sample, validity_mask
from sfn.training.train_mfms import MFMSActorCritic, make_mfms_history_state


def test_tanh_normal_actions_are_bounded_and_log_prob_is_corrected():
    torch.manual_seed(4)
    mean = torch.zeros(128, 3)
    action, log_prob, entropy = tanh_normal_sample(mean, torch.zeros(3))
    assert torch.all(action > -1) and torch.all(action < 1)
    assert action.shape == (128, 3)
    assert log_prob.shape == entropy.shape == (128,)
    raw = torch.atanh(action)
    expected = (
        torch.distributions.Normal(mean, torch.ones_like(mean)).log_prob(raw) - torch.log(1 - action.square() + 1e-6)
    ).sum(-1)
    assert torch.allclose(log_prob, expected, atol=1e-5)


def test_gae_bootstraps_truncation_but_not_termination_and_stops_trace():
    values = [torch.tensor(1.0), torch.tensor(2.0)]
    next_values = [torch.tensor(7.0), torch.tensor(9.0)]
    truncated = compute_gae([0.0, 0.0], values, next_values, [False, False], [True, False], 0.5, 1.0)
    terminated = compute_gae([0.0, 0.0], values, next_values, [True, False], [False, False], 0.5, 1.0)
    assert truncated[0].item() == pytest.approx(2.5)  # .5 * 7 - 1
    assert terminated[0].item() == pytest.approx(-1.0)


def test_left_padding_mask_and_last_valid_selection():
    seq, mask = make_mfms_history_state([torch.ones(452)], 4, "cpu", return_mask=True)
    assert mask.tolist() == [[False, False, False, True]]
    assert last_valid_indices(mask).tolist() == [3]
    assert validity_mask([1, 3], 4, padding="right").tolist() == [
        [True, False, False, False],
        [True, True, True, False],
    ]
    model = MFMSActorCritic(projection_dim=8, hidden_dim=8)
    mean, value, _ = model(seq, valid_mask=mask)
    assert mean.shape == (1, 3) and value.shape == (1,)


def test_checkpoint_additive_metadata_preserves_v1_compatibility(tmp_path):
    path = tmp_path / "checkpoint.pt"
    checkpoint = make_checkpoint("x", {}, {}, epoch=2, global_step=12, run={"seed": 3}, train_config={"gamma": 0.9})
    save_checkpoint(path, checkpoint)
    loaded = load_checkpoint_cpu(path)
    assert loaded["schema_version"] == 1
    assert loaded["run"]["seed"] == 3
    assert loaded["train_config"]["gamma"] == 0.9


def test_checkpoint_runtime_compatibility_rejects_missing_and_mismatch():
    checkpoint = {"compatibility": {"renderer_backend": "mesh_orthographic", "mask_source": "predicted"}}
    expected = {"renderer_backend": "panda_native_camera", "mask_source": "predicted"}
    with pytest.raises(ValueError, match="renderer_backend"):
        assert_checkpoint_compatible(checkpoint, expected)
    with pytest.raises(ValueError, match="position_sha256=missing"):
        assert_checkpoint_compatible(checkpoint, {"position_sha256": "abc"})
    assert_checkpoint_compatible(checkpoint, expected, allow_incompatible=True)
