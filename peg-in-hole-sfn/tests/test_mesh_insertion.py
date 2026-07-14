import numpy as np
from sfn.config import InsertionConfig
from sfn.envs.asset_registry import AssetRegistry
from sfn.envs.mesh_insertion import simulate_mesh_insertion


def test_exact_mesh_insertion_reaches_measured_depth_for_all_shapes():
    config = InsertionConfig(
        collision_mode="geometric",
        target_depth_mm=8.0,
        descent_increment_mm=0.25,
        geometry_pixels_per_mm=20.0,
    )
    for shape in AssetRegistry().list_shapes():
        result = simulate_mesh_insertion(shape, np.zeros(3), config)
        assert result.success, (shape, result)
        assert result.reached_depth
        assert not result.collision_failure
        assert abs(result.insertion_depth_mm - 8.0) < 1e-9
        assert abs((result.peg_tip_start_z_mm - result.peg_tip_final_z_mm) - 8.0) < 1e-6


def test_intentional_mesh_misalignment_hits_rim_before_target_depth():
    config = InsertionConfig(collision_mode="geometric", target_depth_mm=8.0)
    result = simulate_mesh_insertion("square-concave1", np.asarray([0.003, 0.0, 0.0]), config)
    assert not result.success
    assert result.collision_failure
    assert not result.reached_depth
    assert result.outside_pixels > 0
    assert result.penetration_proxy_mm > 0
