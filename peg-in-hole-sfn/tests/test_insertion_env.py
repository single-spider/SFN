import numpy as np
from sfn.envs import InsertionConfig, PegInHoleInsertionEnv


def _zero_action():
    return np.zeros(3, dtype=np.float32)


def test_exact_alignment_inserts_successfully():
    env = PegInHoleInsertionEnv(seed=1, shapes=["synthetic-square"])
    obs, info = env.reset(
        seed=1,
        options={"shape": "synthetic-square", "pose_error": [0.0, 0.0, 0.0]},
    )
    obs, reward, terminated, truncated, info = env.step(_zero_action())
    assert terminated
    assert not truncated
    assert info["success"]
    assert info["phase"] == "SUCCESS"
    assert info["insertion_depth_mm"] == info["target_depth_mm"]
    assert not info["collision_failure"]
    env.close()


def test_alignment_success_but_insertion_tolerance_failure():
    env = PegInHoleInsertionEnv(
        seed=1,
        shapes=["synthetic-square"],
        insertion_config=InsertionConfig(insertion_xy_axis_mm=0.6, insertion_yaw_deg=1.0),
    )
    # 0.8 mm is inside the alignment tolerance of 1.0 mm, so the env descends,
    # but it is outside the insertion tolerance of 0.6 mm and must fail.
    obs, info = env.reset(
        seed=1,
        options={"shape": "synthetic-square", "pose_error": [0.0008, 0.0, 0.0]},
    )
    obs, reward, terminated, truncated, info = env.step(_zero_action())
    assert terminated
    assert not truncated
    assert not info["success"]
    assert info["phase"] == "FAILURE"
    assert not info["insertion_residual_ok"]
    assert info["collision_failure"]
    env.close()


def test_public_attempt_insertion_is_deterministic():
    env = PegInHoleInsertionEnv(seed=7, shapes=["synthetic-square"])
    env.reset(seed=7, options={"shape": "synthetic-square", "pose_error": [0.0, 0.0, 0.0]})
    success1, info1 = env.attempt_insertion()
    env.reset(seed=7, options={"shape": "synthetic-square", "pose_error": [0.0, 0.0, 0.0]})
    success2, info2 = env.attempt_insertion()
    assert success1 == success2
    assert info1["insertion_depth_mm"] == info2["insertion_depth_mm"]
    assert info1["descent_attempts"] == info2["descent_attempts"]
    env.close()
