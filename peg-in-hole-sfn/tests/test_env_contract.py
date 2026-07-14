import numpy as np
from sfn.envs import PegInHoleAlignmentEnv


def test_reset_and_step_contract():
    env = PegInHoleAlignmentEnv(seed=123, shapes=["synthetic-square"])
    obs, info = env.reset(seed=123, options={"pose_error": [0.004, -0.003, 5.0], "shape": "synthetic-square"})
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, info = env.step(np.array([-1.0, 1.0, -1.0], dtype=np.float32))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_oracle_converges():
    env = PegInHoleAlignmentEnv(seed=1, shapes=["synthetic-square"])
    obs, info = env.reset(seed=1, options={"pose_error": [0.004, -0.004, 5.0], "shape": "synthetic-square"})
    for _ in range(env.config.max_steps):
        action = -obs["pose_error"]
        action = np.array(
            [
                action[0] * 1000 / env.config.max_action_xy_mm,
                action[1] * 1000 / env.config.max_action_xy_mm,
                action[2] / env.config.max_action_yaw_deg,
            ],
            dtype=np.float32,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["success"]
    env.close()
