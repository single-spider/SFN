import pytest


def test_final_severity_profiles_are_camera_executable_not_label_edits():
    from sfn.evaluation.disturbance import ROBUSTNESS_PROFILES

    for level in range(5):
        profile = ROBUSTNESS_PROFILES[f"severity_{level}"]
        assert profile.camera_only
        assert profile.seam_dropout_prob == 0
        assert profile.label_flip_prob == 0


def test_episode_keyed_burst_is_consecutive_and_reproducible():
    torch = pytest.importorskip("torch")
    from sfn.evaluation.disturbance import DisturbanceConfig, DisturbedVirtualSensorNetwork
    from sfn.models.vsn import VirtualSensorNetwork

    cfg = DisturbanceConfig(name="test_burst", occlusion_frac=0.25, occlusion_burst_length=3, occlusion_burst_period=6)
    wrapper = DisturbedVirtualSensorNetwork(VirtualSensorNetwork(), cfg)
    mask = torch.ones((1, 40, 40), dtype=torch.long)
    wrapper.set_episode_seed(91)
    first = [wrapper._disturb_mask(mask, frame_index=i).clone() for i in range(6)]
    wrapper.set_episode_seed(91)
    second = [wrapper._disturb_mask(mask, frame_index=i).clone() for i in range(6)]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(int((x == 0).sum()) > 0 for x in first[:3])
    assert all(int((x == 0).sum()) == 0 for x in first[3:])


def test_standalone_pybullet_exact_and_misaligned_square():
    pytest.importorskip("pybullet")
    from sfn.config import InsertionConfig
    from sfn.envs.pybullet_insertion import simulate_pybullet_insertion

    cfg = InsertionConfig(target_depth_mm=3.0, descent_increment_mm=0.2, max_descent_attempts=20)
    exact = simulate_pybullet_insertion("square-square", [0, 0, 0], cfg)
    bad = simulate_pybullet_insertion("square-square", [0.002, 0, 0], cfg)
    assert exact.success and exact.measured_depth_mm >= 3.0 - 0.01
    assert not bad.success and bad.reason == "rim_collision"


def test_standalone_pybullet_normalizes_legacy_diamond_asset_frame():
    pytest.importorskip("pybullet")
    from sfn.config import InsertionConfig
    from sfn.envs.pybullet_insertion import simulate_pybullet_insertion

    cfg = InsertionConfig(target_depth_mm=3.0, descent_increment_mm=0.2, max_descent_attempts=20)
    exact = simulate_pybullet_insertion("square-diamond", [0, 0, 0], cfg)
    bad = simulate_pybullet_insertion("square-diamond", [0.002, 0, 0], cfg)
    assert exact.success
    assert not bad.success and bad.reason == "rim_collision"


def test_panda_insertion_alignment_uses_insertion_tolerance():
    pytest.importorskip("pybullet")
    from sfn.config import InsertionConfig
    from sfn.panda.panda_insertion_env import PandaPegInHoleInsertionEnv

    insertion = InsertionConfig(insertion_xy_axis_mm=0.6, insertion_yaw_deg=1.0)
    env = PandaPegInHoleInsertionEnv(shapes=["square-square"], insertion_config=insertion)
    try:
        assert env.config.xy_success_axis_mm == insertion.insertion_xy_axis_mm
        assert env.config.yaw_success_deg == insertion.insertion_yaw_deg
        assert env.config.task == "insertion"
    finally:
        env.close()


def test_geometric_position_checkpoint_loader_supports_calibrated_model():
    from pathlib import Path

    from sfn.evaluation.evaluate_perception import _load_model

    checkpoint = Path("models/position_mesh_v2_geometric.pt")
    if not checkpoint.exists():
        pytest.skip("optional trained checkpoint absent")
    model = _load_model("position", checkpoint)
    assert hasattr(model, "predict_continuous")
