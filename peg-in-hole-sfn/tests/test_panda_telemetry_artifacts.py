from __future__ import annotations

import csv
import json

import numpy as np
from sfn.config import EnvironmentConfig, InsertionConfig
from sfn.panda import PandaConfig, PandaPegInHoleAlignmentEnv, PandaPegInHoleInsertionEnv
from sfn.panda.artifacts import panda_per_shape_rows, write_panda_per_shape


def test_dynamic_panda_step_exposes_command_joint_ik_and_contact_telemetry():
    env = PandaPegInHoleAlignmentEnv(
        shapes=["square-concave1"],
        panda_config=PandaConfig(execution_mode="dynamic", command_steps=2),
    )
    try:
        env.reset(
            seed=41,
            options={"shape": "square-concave1", "pose_error": [0.001, -0.001, 1.0], "nontrivial": False},
        )
        _obs, _reward, _terminated, _truncated, info = env.step([0.25, -0.5, 0.75])
        assert info["action_physical"].shape == (3,)
        assert info["measured_action_physical"].shape == (3,)
        assert info["joint_positions"].shape == (7,)
        assert info["joint_target"].shape == (7,)
        assert info["joint_limit_margins"].shape == (7,)
        assert np.isclose(info["min_joint_limit_margin"], np.min(info["joint_limit_margins"]))
        assert info["ik_branch"] == "measured_rest"
        assert info["ik_residual_m"] is None
        assert info["contact_count"] == len(info["contacts"])
        assert info["max_contact_force"] >= 0.0
        assert info["max_penetration_mm"] >= 0.0
        assert info["failure_state"] in {None, "panda_alignment_timeout"}
    finally:
        env.close()


def test_panda_insertion_trace_keeps_each_descent_attempt_telemetry():
    env = PandaPegInHoleInsertionEnv(
        shapes=["square-concave1"],
        env_config=EnvironmentConfig(xy_success_axis_mm=1.0, yaw_success_deg=2.0),
        panda_config=PandaConfig(execution_mode="kinematic"),
        insertion_config=InsertionConfig(descent_increment_mm=0.5, target_depth_mm=1.0, max_descent_attempts=3),
    )
    try:
        env.reset(seed=42, options={"shape": "square-concave1", "pose_error": [0, 0, 0], "nontrivial": False})
        _obs, _reward, terminated, _truncated, info = env.step([0, 0, 0])
        assert terminated
        assert len(info["insertion_trace"]) == info["insertion_attempts"]
        attempt = info["insertion_trace"][0]
        required = {
            "commanded_dz_m",
            "measured_dz_m",
            "joint_positions",
            "joint_limit_margins",
            "ik_residual_m",
            "ik_branch",
            "contacts",
            "max_contact_force",
            "max_penetration_mm",
            "lateral_drift_mm",
            "insertion_depth_mm",
            "failure_state",
        }
        assert required <= attempt.keys()
    finally:
        env.close()


def test_panda_per_shape_csv_preserves_method_metrics_and_failure_states(tmp_path):
    records = [
        {
            "shape": "shape-a",
            "method": "oracle",
            "success": True,
            "steps": 1,
            "final_xy_error_mm": 0.1,
            "insertion_depth_mm": 3.0,
            "min_joint_limit_margin": 0.2,
        },
        {
            "shape": "shape-a",
            "method": "oracle",
            "success": False,
            "steps": 2,
            "failure_state": "panda_insertion_jam",
            "final_xy_error_mm": 0.3,
            "insertion_depth_mm": 1.0,
            "min_joint_limit_margin": 0.1,
        },
    ]
    rows = panda_per_shape_rows(records)
    assert rows[0]["episodes"] == 2
    assert rows[0]["success_rate"] == 0.5
    assert rows[0]["failure_states"] == {"panda_insertion_jam": 1}
    assert rows[0]["min_joint_limit_margin"] == 0.1

    output = write_panda_per_shape(tmp_path / "per_shape.csv", records)
    with output.open(newline="", encoding="utf-8") as stream:
        persisted = next(csv.DictReader(stream))
    assert persisted["shape"] == "shape-a"
    assert json.loads(persisted["failure_states"]) == {"panda_insertion_jam": 1}
