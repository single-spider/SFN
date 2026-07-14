"""Validated dataclass configuration and lightweight YAML loading."""

from __future__ import annotations

import ast
import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .constants import ORIENTATION_ANGLES_DEG, POSITION_GRID_SIZE


@dataclass
class ProjectConfig:
    seed: int = 1
    device: str = "auto"
    output_root: str = "artifacts"


@dataclass
class EnvironmentConfig:
    task: str = "alignment"
    max_steps: int = 20
    xy_initial_range_mm: float = 10.0
    yaw_initial_range_deg: float = 10.0
    xy_success_axis_mm: float = 1.0
    yaw_success_deg: float = 2.0
    xy_workspace_mm: float = 15.0
    yaw_workspace_deg: float = 15.0
    max_action_xy_mm: float = 2.0
    max_action_yaw_deg: float = 2.0
    reward_mode: str = "dense_progress"
    gui: bool = False
    reward_w_xy: float = 0.7
    reward_w_yaw: float = 0.3
    step_penalty: float = 0.01
    success_bonus: float = 1.0
    failure_penalty: float = 1.0


@dataclass
class CameraConfig:
    render_width: int = 1280
    render_height: int = 720
    crop_width: int = 250
    crop_height: int = 200
    fov_y_deg: float = 45.0
    near: float = 0.001
    far: float = 10.0
    eye_offset: tuple[float, float, float] = (0.0, 0.1, 0.1)
    up: tuple[float, float, float] = (0.0, -1.0, 0.0)
    # ``toy_direct`` preserves the original fast rectangle renderer for legacy
    # checkpoints and unit tests. ``mesh_orthographic`` rasterizes the actual
    # peg and hole-opening mesh assets and is the minimum valid backend for
    # unseen-shape research claims.
    renderer_backend: str = "toy_direct"
    orthographic_pixels_per_mm: float = 4.0


@dataclass
class VSNConfig:
    position_grid_size: int = POSITION_GRID_SIZE
    position_resolution_mm: float = 1.0
    orientation_angles_deg: list[float] = field(default_factory=lambda: list(ORIENTATION_ANGLES_DEG))
    mask_source: str = "ground_truth"


@dataclass
class EvaluationConfig:
    episodes_per_shape: int = 100
    save_videos: bool = False


@dataclass
class InsertionConfig:
    descent_increment_mm: float = 0.25
    target_depth_mm: float = 8.0
    max_descent_attempts: int = 64
    collision_mode: str = "geometric"
    insertion_xy_axis_mm: float = 0.6
    insertion_yaw_deg: float = 1.0
    max_collision_penetration_mm: float = 0.0
    approach_clearance_mm: float = 0.5
    geometry_pixels_per_mm: float = 20.0


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vsn: VSNConfig = field(default_factory=VSNConfig)
    insertion: InsertionConfig = field(default_factory=InsertionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        _require_non_negative_integer("project.seed", self.project.seed)

        if self.environment.task not in {"alignment", "insertion"}:
            raise ValueError("environment.task must be alignment or insertion")
        if self.environment.reward_mode not in {"dense_progress", "paper"}:
            raise ValueError("environment.reward_mode must be dense_progress or paper")
        _require_positive_integer("environment.max_steps", self.environment.max_steps)
        for name in ("xy_initial_range_mm", "yaw_initial_range_deg"):
            _require_non_negative(f"environment.{name}", getattr(self.environment, name))
        for name in (
            "xy_success_axis_mm",
            "yaw_success_deg",
            "xy_workspace_mm",
            "yaw_workspace_deg",
            "max_action_xy_mm",
            "max_action_yaw_deg",
        ):
            _require_positive(f"environment.{name}", getattr(self.environment, name))
        for name in ("reward_w_xy", "reward_w_yaw", "step_penalty", "success_bonus", "failure_penalty"):
            _require_non_negative(f"environment.{name}", getattr(self.environment, name))
        if self.environment.reward_w_xy == 0 and self.environment.reward_w_yaw == 0:
            raise ValueError("environment reward weights must not both be zero")
        if self.environment.xy_workspace_mm < self.environment.xy_initial_range_mm:
            raise ValueError("environment.xy_workspace_mm must cover environment.xy_initial_range_mm")
        if self.environment.yaw_workspace_deg < self.environment.yaw_initial_range_deg:
            raise ValueError("environment.yaw_workspace_deg must cover environment.yaw_initial_range_deg")

        for name in ("render_width", "render_height", "crop_width", "crop_height"):
            _require_positive_integer(f"camera.{name}", getattr(self.camera, name))
        if self.camera.crop_width > self.camera.render_width or self.camera.crop_height > self.camera.render_height:
            raise ValueError("camera crop dimensions must not exceed render dimensions")
        _require_range("camera.fov_y_deg", self.camera.fov_y_deg, lower=0.0, upper=180.0)
        _require_positive("camera.near", self.camera.near)
        _require_positive("camera.far", self.camera.far)
        if self.camera.far <= self.camera.near:
            raise ValueError("camera.far must be greater than camera.near")
        if self.camera.renderer_backend not in {"toy_direct", "mesh_orthographic"}:
            raise ValueError("camera.renderer_backend must be toy_direct or mesh_orthographic")
        _require_positive("camera.orthographic_pixels_per_mm", self.camera.orthographic_pixels_per_mm)

        _require_positive_integer("vsn.position_grid_size", self.vsn.position_grid_size)
        _require_positive("vsn.position_resolution_mm", self.vsn.position_resolution_mm)
        if not isinstance(self.vsn.orientation_angles_deg, (list, tuple)) or not self.vsn.orientation_angles_deg:
            raise ValueError("vsn.orientation_angles_deg must be a non-empty sequence")
        for index, angle in enumerate(self.vsn.orientation_angles_deg):
            _require_finite(f"vsn.orientation_angles_deg[{index}]", angle)
        if self.vsn.mask_source not in {"ground_truth", "predicted"}:
            raise ValueError("vsn.mask_source must be ground_truth or predicted")
        grid_half_mm = (self.vsn.position_grid_size - 1) * self.vsn.position_resolution_mm / 2.0
        if grid_half_mm < self.environment.xy_initial_range_mm:
            raise ValueError("VSN grid cannot cover initial XY range")

        if self.insertion.collision_mode not in {"proxy", "geometric"}:
            raise ValueError("insertion.collision_mode must be proxy or geometric")
        for name in (
            "descent_increment_mm",
            "target_depth_mm",
            "insertion_xy_axis_mm",
            "insertion_yaw_deg",
            "geometry_pixels_per_mm",
        ):
            _require_positive(f"insertion.{name}", getattr(self.insertion, name))
        _require_positive_integer("insertion.max_descent_attempts", self.insertion.max_descent_attempts)
        for name in ("max_collision_penetration_mm", "approach_clearance_mm"):
            _require_non_negative(f"insertion.{name}", getattr(self.insertion, name))

        _require_positive_integer("evaluation.episodes_per_shape", self.evaluation.episodes_per_shape)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_scalar(text: str) -> Any:
    text = text.strip()
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text.strip("\"'")


def _fallback_yaml_load(raw: str) -> dict[str, Any]:
    root = {}
    stack = [(-1, root)]
    for original in raw.splitlines():
        line = original.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _load_yaml_file(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml

        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping in {path}")
        return loaded
    except ModuleNotFoundError:
        return _fallback_yaml_load(raw)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in update.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _require_finite(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")


def _require_positive(name: str, value: Any) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: Any) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_range(name: str, value: Any, *, lower: float, upper: float) -> None:
    _require_finite(name, value)
    if not lower < value < upper:
        raise ValueError(f"{name} must be greater than {lower:g} and less than {upper:g}")


def _strict_section(data: dict[str, Any], name: str, cls: type[Any]) -> Any:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section {name!r} must be a mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(value) - known)
    if unknown:
        paths = ", ".join(f"{name}.{key}" for key in unknown)
        raise ValueError(f"Unknown configuration key(s): {paths}")
    return cls(**value)


def config_from_dict(data: dict[str, Any]) -> Config:
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    known_sections = {item.name for item in fields(Config)}
    unknown_sections = sorted(set(data) - known_sections)
    if unknown_sections:
        raise ValueError(f"Unknown configuration key(s): {', '.join(unknown_sections)}")
    cfg = Config(
        project=_strict_section(data, "project", ProjectConfig),
        environment=_strict_section(data, "environment", EnvironmentConfig),
        camera=_strict_section(data, "camera", CameraConfig),
        vsn=_strict_section(data, "vsn", VSNConfig),
        insertion=_strict_section(data, "insertion", InsertionConfig),
        evaluation=_strict_section(data, "evaluation", EvaluationConfig),
    )
    cfg.validate()
    return cfg


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    result = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value: {item!r}")
        key, raw = item.split("=", 1)
        target = result
        for part in key.split(".")[:-1]:
            target = target.setdefault(part, {})
        target[key.split(".")[-1]] = _coerce_scalar(raw)
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None, seed: int | None = None) -> Config:
    data = Config().to_dict()
    if path is not None:
        data = _merge(data, _load_yaml_file(Path(path)))
    data = _merge(data, parse_overrides(overrides))
    if seed is not None:
        data.setdefault("project", {})["seed"] = seed
    return config_from_dict(data)


def dump_resolved_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
