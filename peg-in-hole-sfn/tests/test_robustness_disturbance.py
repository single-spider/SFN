import numpy as np
import torch
from sfn.evaluation.disturbance import (
    DisturbanceConfig,
    EnsembleVirtualSensorNetwork,
    TemporalSmoothedVirtualSensorNetwork,
    _shift_zero_fill,
    disturb_observation,
    parse_profile_names,
)
from sfn.models.vsn import VirtualSensorNetwork, VSNOutput


class _CountingVSN(VirtualSensorNetwork):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, rgb=None, mask=None):
        self.calls += 1
        pos = torch.zeros((1, 21, 21), dtype=torch.float32)
        pos[0, 10, 10] = 1.0
        ori = torch.zeros((1, 11), dtype=torch.float32)
        ori[0, 5] = 1.0
        return VSNOutput(
            mask_logits=None,
            mask=torch.zeros((1, 4, 4), dtype=torch.long),
            position_logits=torch.zeros((1, 441), dtype=torch.float32),
            position_prob=pos,
            orientation_scores=torch.zeros((1, 11), dtype=torch.float32),
            orientation_prob=ori,
            dxy_m=torch.zeros((1, 2), dtype=torch.float32),
            dyaw_deg=torch.zeros((1,), dtype=torch.float32),
            position_confidence=torch.ones((1,), dtype=torch.float32),
            orientation_confidence=torch.ones((1,), dtype=torch.float32),
        )


def test_shift_zero_fill_does_not_wrap():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 1] = 2
    shifted = _shift_zero_fill(mask, dx=2, dy=1)
    assert shifted[2, 3] == 2
    assert shifted[1, 1] == 0
    assert shifted.sum() == 2


def test_parse_profile_names_rejects_unknown():
    assert parse_profile_names("clean,combined") == ["clean", "combined"]
    try:
        parse_profile_names("clean,nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_temporal_wrapper_calls_wrapped_vsn():
    base = _CountingVSN()
    wrapped = TemporalSmoothedVirtualSensorNetwork(base, alpha=0.8)
    wrapped(mask=torch.zeros((1, 4, 4), dtype=torch.long))
    assert base.calls == 1


def test_ensemble_wrapper_calls_wrapped_vsn_for_each_sample():
    base = _CountingVSN()
    wrapped = EnsembleVirtualSensorNetwork(base, samples=3)
    wrapped(mask=torch.zeros((1, 4, 4), dtype=torch.long))
    assert base.calls == 3


def test_observation_disturbance_is_episode_frame_keyed_and_changes_camera_data():
    rgb = np.full((3, 20, 24), 128, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[4:10, 5:11] = 1
    mask[10:14, 8:15] = 2
    config = DisturbanceConfig(rgb_noise_std=0.1, mask_shift_px=2, occlusion_prob=1.0, occlusion_frac=0.2)
    first = disturb_observation({"rgb": rgb, "mask": mask}, config, episode_seed=7, frame_index=3)
    second = disturb_observation({"rgb": rgb, "mask": mask}, config, episode_seed=7, frame_index=3)
    other = disturb_observation({"rgb": rgb, "mask": mask}, config, episode_seed=7, frame_index=4)
    assert np.array_equal(first["rgb"], second["rgb"])
    assert np.array_equal(first["mask"], second["mask"])
    assert not np.array_equal(first["rgb"], rgb)
    assert not np.array_equal(first["rgb"], other["rgb"])
