import json

import numpy as np
import pytest
from sfn.data.augment import (
    DomainRandomizationConfig,
    apply_domain_randomization,
    replay_domain_randomization,
)


def _sample():
    rgb = np.full((3, 48, 64), 110, dtype=np.uint8)
    rgb[0, 8:40, 12:52] = 210
    rgb[1, 16:32, 20:44] = 40
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[8:40, 12:52] = 1
    mask[16:32, 20:44] = 2
    depth = np.zeros((48, 64), dtype=np.float32)
    depth[mask > 0] = 0.24
    return rgb, mask, depth


def test_expanded_families_are_deterministic_json_records_and_preserve_classes():
    rgb, mask, depth = _sample()
    first = apply_domain_randomization(rgb, mask, "heavy", seed=90210, depth=depth, return_record=True)
    second = apply_domain_randomization(rgb, mask, "heavy", seed=90210, depth=depth, return_record=True)
    for left, right in zip(first[:3], second[:3], strict=True):
        assert np.array_equal(left, right)
    assert first[3] == second[3]
    json.dumps(first[3])
    assert set(np.unique(first[1])) <= set(np.unique(mask))
    assert first[1].dtype == mask.dtype
    assert first[2].dtype == depth.dtype and np.all(first[2] >= 0)
    expected = {
        "color_illumination_gradient", "material_texture_background", "shadow", "shot_noise",
        "motion_blur", "gamma", "white_balance", "seam", "depth_noise",
    }
    assert expected <= first[3].keys()


def test_record_replays_rgb_mask_and_optional_depth_exactly():
    rgb, mask, depth = _sample()
    out_rgb, out_mask, out_depth, record = apply_domain_randomization(
        rgb, mask, "medium", rng=np.random.default_rng(17), depth=depth, return_record=True
    )
    replay_rgb, replay_mask, replay_depth = replay_domain_randomization(rgb, mask, record, depth=depth)
    assert np.array_equal(replay_rgb, out_rgb)
    assert np.array_equal(replay_mask, out_mask)
    assert np.array_equal(replay_depth, out_depth)


def test_config_contract_is_strict_additive_and_depth_noise_requires_depth_output():
    rgb, mask, depth = _sample()
    config = DomainRandomizationConfig(
        color_illumination_gradient=False, material_texture_background=False, shadow=False,
        shot_noise=False, motion_blur=False, gamma=False, white_balance=False, seam=False, depth_noise=True,
    )
    _, _, record = apply_domain_randomization(rgb, mask, "heavy", seed=3, config=config, return_record=True)
    assert record["depth_noise"]["enabled"] is False
    assert all(not record[name]["enabled"] for name in (
        "color_illumination_gradient", "material_texture_background", "shadow", "shot_noise",
        "motion_blur", "gamma", "white_balance", "seam",
    ))
    _, _, noisy_depth, depth_record = apply_domain_randomization(
        rgb, mask, "heavy", seed=3, config=config, depth=depth, return_record=True
    )
    assert depth_record["depth_noise"]["enabled"] is True
    assert not np.array_equal(noisy_depth, depth)
    with pytest.raises(ValueError, match="Unknown domain-randomization config key"):
        apply_domain_randomization(rgb, mask, config={"shdaow": True})
    with pytest.raises(ValueError, match="must be boolean"):
        apply_domain_randomization(rgb, mask, config={"gamma": 1})


def test_none_remains_identity_for_rgb_mask_and_depth():
    rgb, mask, depth = _sample()
    out_rgb, out_mask, out_depth, record = apply_domain_randomization(
        rgb, mask, seed=1, depth=depth, return_record=True
    )
    assert np.array_equal(out_rgb, rgb)
    assert np.array_equal(out_mask, mask)
    assert np.array_equal(out_depth, depth)
    assert record["version"] == 1 and record["level"] == "none"
