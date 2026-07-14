from __future__ import annotations

from sfn.config import EnvironmentConfig, InsertionConfig
from sfn.geometry import physical_to_normalized_action
from sfn.panda import PandaConfig, PandaPegInHoleInsertionEnv


def _env(**insertion_kwargs):
    return PandaPegInHoleInsertionEnv(
        shapes=["square-concave1"],
        env_config=EnvironmentConfig(xy_success_axis_mm=1.0, yaw_success_deg=2.0),
        panda_config=PandaConfig(execution_mode="kinematic"),
        insertion_config=InsertionConfig(**insertion_kwargs),
    )


def test_panda_kinematic_insertion_uses_incremental_measured_tip_depth():
    env = _env(descent_increment_mm=1.0, target_depth_mm=3.0, max_descent_attempts=6)
    try:
        env.reset(seed=3, options={"shape": "square-concave1", "pose_error": [0, 0, 0], "nontrivial": False})
        _, _, terminated, truncated, info = env.step([0, 0, 0])
        assert terminated and not truncated
        assert info["insertion_success"]
        assert info["termination_reason"] == "panda_insertion_success"
        assert info["insertion_depth_mm"] >= 3.0
        assert 1 <= info["insertion_attempts"] <= 6
        assert info["execution_mode"] == "kinematic"
        assert info["lateral_drift_mm"] <= env.insertion_config.insertion_xy_axis_mm
    finally:
        env.close()


def test_panda_insertion_continues_until_strict_insertion_tolerance():
    env = _env(descent_increment_mm=1.0, target_depth_mm=2.0, max_descent_attempts=3, insertion_xy_axis_mm=0.1)
    try:
        env.reset(seed=4, options={"shape": "square-concave1", "pose_error": [0.0005, 0, 0], "nontrivial": False})
        obs, _, terminated, truncated, info = env.step([0, 0, 0])
        assert not terminated and not truncated
        assert not info["insertion_success"]
        action = physical_to_normalized_action(
            [-obs["pose_error"][0], -obs["pose_error"][1], -obs["pose_error"][2]],
            env.config.max_action_xy_mm,
            env.config.max_action_yaw_deg,
        )
        _, _, terminated, truncated, info = env.step(action)
        assert terminated and not truncated
        assert info["insertion_success"]
        assert info["termination_reason"] == "panda_insertion_success"
    finally:
        env.close()


def test_panda_insertion_timeout_reason_is_explicit():
    env = _env(descent_increment_mm=0.25, target_depth_mm=20.0, max_descent_attempts=1)
    try:
        env.reset(seed=5, options={"shape": "square-concave1", "pose_error": [0, 0, 0], "nontrivial": False})
        _, _, terminated, truncated, info = env.step([0, 0, 0])
        assert terminated and not truncated
        assert not info["insertion_success"]
        assert info["insertion_attempts"] == 1
        assert info["termination_reason"] == "panda_insertion_timeout"
    finally:
        env.close()


def test_insertion_observer_replaces_duplicate_descent_substep_samples():
    motion_samples = []
    insertion_samples = []
    env = PandaPegInHoleInsertionEnv(
        shapes=["square-concave1"],
        panda_config=PandaConfig(execution_mode="kinematic"),
        insertion_config=InsertionConfig(descent_increment_mm=1.0, target_depth_mm=3.0, max_descent_attempts=6),
        motion_observer=lambda *_args: motion_samples.append(1),
        motion_observer_stride=1,
        insertion_observer=lambda _obs, trace: insertion_samples.append(trace["insertion_depth_mm"]),
    )
    try:
        env.reset(seed=6, options={"shape": "square-concave1", "pose_error": [0, 0, 0], "nontrivial": False})
        _, _, terminated, truncated, info = env.step([0, 0, 0])
        assert terminated and not truncated and info["insertion_success"]
        assert len(insertion_samples) == info["insertion_attempts"]
        assert insertion_samples == sorted(insertion_samples)
        # The ordinary alignment command contributes one 120-substep motor
        # trajectory. Descent attempts do not add another 120 samples each.
        assert len(motion_samples) == 120
    finally:
        env.close()
