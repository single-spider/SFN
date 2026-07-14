from __future__ import annotations

from sfn.geometry import physical_to_normalized_action
from sfn.panda import PandaPegInHoleAlignmentEnv
from sfn.panda.config import PandaConfig
from sfn.panda.native_vsn import PandaBodyIdGeometricVSN


def test_panda_alignment_env_oracle_step_succeeds():
    env = PandaPegInHoleAlignmentEnv(shapes=["square-concave1"])
    try:
        obs, info = env.reset(seed=7, options={"pose_error": [0.001, -0.001, 1.0], "shape": "square-concave1"})
        action = physical_to_normalized_action(
            [-obs["pose_error"][0], -obs["pose_error"][1], -obs["pose_error"][2]],
            env.config.max_action_xy_mm,
            env.config.max_action_yaw_deg,
        )
        _obs, _reward, terminated, truncated, info = env.step(action)
        assert terminated
        assert not truncated
        assert info["success"]
        assert info["tracking_error_mm"] <= 0.30
        assert info["tracking_yaw_error_deg"] <= 0.30
    finally:
        env.close()


def test_panda_alignment_env_native_camera_body_id_mask():
    env = PandaPegInHoleAlignmentEnv(shapes=["square-concave1"], panda_config=PandaConfig(native_camera=True))
    try:
        obs, _info = env.reset(seed=8, options={"pose_error": [0.001, -0.001, 1.0], "shape": "square-concave1"})
        assert obs["rgb"].shape == (3, env.camera_config.crop_height, env.camera_config.crop_width)
        assert obs["mask"].shape == (env.camera_config.crop_height, env.camera_config.crop_width)
        assert (obs["mask"] == 1).sum() > 0
        assert (obs["mask"] == 2).sum() > 0
    finally:
        env.close()


def test_panda_native_geometric_vsn_returns_policy_state():
    env = PandaPegInHoleAlignmentEnv(shapes=["square-concave1"], panda_config=PandaConfig(native_camera=True))
    try:
        obs, _info = env.reset(seed=9, options={"pose_error": [0.001, -0.001, 1.0], "shape": "square-concave1"})
        import torch

        out = PandaBodyIdGeometricVSN()(mask=torch.as_tensor(obs["mask"][None], dtype=torch.long))
        assert out.position_prob.shape == (1, 21, 21)
        assert out.orientation_prob.shape == (1, 11)
        assert float(out.position_prob.sum()) == 1.0
        assert float(out.orientation_prob.sum()) == 1.0
    finally:
        env.close()
