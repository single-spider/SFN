"""Strict configuration loading and boundary validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sfn.config import config_from_dict, load_config
from sfn.evaluation.disturbance import ActionDisturbanceConfig, DisturbanceConfig
from sfn.training.curriculum import stages_from_mapping

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("section", "unknown"),
    [
        ("project", "sead"),
        ("environment", "max_step"),
        ("camera", "render_wdth"),
        ("vsn", "grid_size"),
        ("insertion", "target_dept_mm"),
        ("evaluation", "episode_per_shape"),
    ],
)
def test_unknown_keys_fail_with_the_full_section_path(section: str, unknown: str) -> None:
    with pytest.raises(ValueError, match=rf"Unknown configuration key.*{section}\.{unknown}"):
        config_from_dict({section: {unknown: 1}})


def test_unknown_top_level_key_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match=r"Unknown configuration key.*evalution"):
        config_from_dict({"evalution": {}})


def test_unknown_override_key_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"Unknown configuration key.*camera\.widht"):
        load_config(overrides=["camera.widht=640"])


def test_section_must_be_a_mapping() -> None:
    with pytest.raises(ValueError, match=r"section 'camera' must be a mapping"):
        config_from_dict({"camera": 640})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"project": {"seed": -1}}, r"project\.seed.*non-negative integer"),
        ({"environment": {"max_steps": 0}}, r"environment\.max_steps.*positive integer"),
        ({"environment": {"xy_initial_range_mm": -0.1}}, r"xy_initial_range_mm.*non-negative"),
        ({"environment": {"max_action_xy_mm": 0}}, r"max_action_xy_mm.*positive"),
        ({"camera": {"render_width": 0}}, r"camera\.render_width.*positive integer"),
        ({"camera": {"fov_y_deg": 180}}, r"camera\.fov_y_deg.*less than 180"),
        ({"camera": {"near": 1.0, "far": 1.0}}, r"camera\.far must be greater"),
        ({"vsn": {"position_grid_size": 0}}, r"position_grid_size.*positive integer"),
        ({"vsn": {"position_resolution_mm": 0}}, r"position_resolution_mm.*positive"),
        ({"vsn": {"orientation_angles_deg": []}}, r"orientation_angles_deg.*non-empty"),
        ({"insertion": {"target_depth_mm": 0}}, r"target_depth_mm.*positive"),
        ({"insertion": {"max_collision_penetration_mm": -0.1}}, r"max_collision_penetration_mm.*non-negative"),
        ({"evaluation": {"episodes_per_shape": 0}}, r"episodes_per_shape.*positive integer"),
    ],
)
def test_meaningful_numeric_boundaries_are_rejected(data: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_dict(data)


def test_non_finite_numeric_value_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"camera\.near.*finite number"):
        config_from_dict({"camera": {"near": float("nan")}})


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda path: path.name)
def test_checked_in_configs_remain_valid(path: Path) -> None:
    if path.name == "sfms_curriculum.yaml":
        stages_from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    elif path.name == "robustness_disturbance_families.yaml":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, values in data["camera"].items():
            DisturbanceConfig(name=name, camera_only=True, **values)
        for name, values in data["action"].items():
            ActionDisturbanceConfig(name=name, **values)
    else:
        load_config(path)
