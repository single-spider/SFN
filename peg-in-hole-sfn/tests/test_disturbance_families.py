import numpy as np
import pytest
from sfn.evaluation.disturbance import (
    ACTION_DISTURBANCE_PROFILES,
    ROBUSTNESS_PROFILES,
    ActionDisturbance,
    ActionDisturbanceConfig,
    DisturbedActionEnv,
    disturb_action,
    disturb_observation,
)


def _camera_fixture():
    yy, xx = np.mgrid[:24, :32]
    rgb = np.stack(((xx * 7) % 256, (yy * 11) % 256, ((xx + yy) * 5) % 256)).astype(np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:19, 7:25] = 1
    mask[10:14, 12:20] = 2
    return {"rgb": rgb, "mask": mask}


@pytest.mark.parametrize(
    "profile_name",
    ["blur", "motion_blur", "gamma", "white_balance", "intrinsic_scale", "crop", "lighting"],
)
def test_camera_family_is_deterministic_changes_rgb_and_never_edits_labels(profile_name):
    observation = _camera_fixture()
    profile = ROBUSTNESS_PROFILES[profile_name]
    first = disturb_observation(observation, profile, episode_seed=82, frame_index=4)
    second = disturb_observation(observation, profile, episode_seed=82, frame_index=4)

    assert profile.camera_only
    assert np.array_equal(first["rgb"], second["rgb"])
    assert not np.array_equal(first["rgb"], observation["rgb"])
    assert np.array_equal(first["mask"], observation["mask"])


def test_camera_randomized_families_are_episode_frame_keyed():
    observation = _camera_fixture()
    for profile_name in ("motion_blur", "crop", "lighting"):
        profile = ROBUSTNESS_PROFILES[profile_name]
        first = disturb_observation(observation, profile, episode_seed=9, frame_index=1)
        other_frame = disturb_observation(observation, profile, episode_seed=9, frame_index=2)
        assert not np.array_equal(first["rgb"], other_frame["rgb"])


def test_final_severity_profiles_keep_values_and_are_label_edit_free():
    expected = (
        (0.0, 0, 0.0, 0.0),
        (0.02, 2, 0.2, 0.08),
        (0.04, 3, 0.4, 0.16),
        (0.07, 5, 0.7, 0.28),
        (0.1, 8, 1.0, 0.4),
    )
    observation = _camera_fixture()
    for level, values in enumerate(expected):
        profile = ROBUSTNESS_PROFILES[f"severity_{level}"]
        assert profile.mask_shift_px == values[1]
        assert (profile.rgb_noise_std, profile.occlusion_prob, profile.occlusion_frac) == pytest.approx(
            (values[0], values[2], values[3])
        )
        assert profile.camera_only
        assert profile.seam_dropout_prob == 0
        assert profile.label_flip_prob == 0
        result = disturb_observation(observation, profile, episode_seed=101, frame_index=7)
        assert np.array_equal(result["mask"], observation["mask"])


def test_camera_only_vsn_mask_input_is_passed_through_without_label_edits():
    torch = pytest.importorskip("torch")
    from sfn.evaluation.disturbance import DisturbedVirtualSensorNetwork
    from sfn.models.vsn import VirtualSensorNetwork

    mask = torch.from_numpy(_camera_fixture()["mask"]).unsqueeze(0).long()
    for level in range(5):
        wrapper = DisturbedVirtualSensorNetwork(VirtualSensorNetwork(), ROBUSTNESS_PROFILES[f"severity_{level}"])
        output = wrapper(mask=mask)
        assert torch.equal(output.mask, mask)


def test_action_helper_is_keyed_and_applies_offsets_backlash_and_clipping():
    config = ActionDisturbanceConfig(
        noise_std=(0.1, 0.1, 0.1),
        backlash=(0.05, 0.05, 0.05),
        calibration_offset=(0.1, 0.0, 0.0),
        attachment_offset=(0.0, -0.1, 0.0),
    )
    action = np.array([0.25, -0.25, 1.0], dtype=np.float32)
    first = disturb_action(action, config, episode_seed=12, frame_index=3)
    second = disturb_action(action, config, episode_seed=12, frame_index=3)
    other = disturb_action(action, config, episode_seed=12, frame_index=4)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.all(np.abs(first) <= 1.0)


def test_action_delay_replays_earlier_disturbed_commands_and_resets():
    config = ActionDisturbanceConfig(delay_steps=2, calibration_offset=(0.1, 0.0, 0.0))
    channel = ActionDisturbance(config)
    channel.set_episode_seed(4)
    outputs = [channel.apply([value, 0.0, 0.0]) for value in (0.1, 0.2, 0.3)]
    assert np.array_equal(outputs[0], np.zeros(3, dtype=np.float32))
    assert np.array_equal(outputs[1], np.zeros(3, dtype=np.float32))
    assert outputs[2][0] == pytest.approx(0.2)
    channel.set_episode_seed(4)
    assert np.array_equal(channel.apply([0.1, 0.0, 0.0]), outputs[0])


class _RecordingEnv:
    def __init__(self):
        self.actions = []

    def reset(self, *, seed=None, options=None):
        self.actions.clear()
        return {"seed": seed}, {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return {}, 0.0, False, False, {"base": True}


def test_action_environment_wrapper_records_commanded_and_executed_actions():
    base = _RecordingEnv()
    wrapped = DisturbedActionEnv(base, ACTION_DISTURBANCE_PROFILES["command_delay"])
    wrapped.reset(seed=5)
    _, _, _, _, info = wrapped.step(np.ones(3, dtype=np.float32))
    assert np.array_equal(base.actions[-1], np.zeros(3, dtype=np.float32))
    assert np.array_equal(info["action_commanded_normalized"], np.ones(3, dtype=np.float32))
    assert np.array_equal(info["action_executed_normalized"], np.zeros(3, dtype=np.float32))
    assert info["action_disturbance"]["delay_steps"] == 2
