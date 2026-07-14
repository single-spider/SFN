"""Shared contracts for controller evaluation runs.

The public CLI treats ``episodes`` as a total run budget.  The legacy
``episodes_per_shape`` keyword remains supported by the individual evaluators,
but is converted to an explicit total before a common, deterministic schedule
is created.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .statistics import percentile_summary, wilson_interval


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    shape: str
    shape_episode: int
    seed: int


def episode_schedule(shapes: Sequence[str], episodes: int, seed: int) -> list[EpisodeSpec]:
    """Return a balanced schedule with seeds shared by every method."""
    shape_list = [str(shape) for shape in shapes]
    if not shape_list:
        raise ValueError("at least one shape is required")
    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    seen: Counter[str] = Counter()
    schedule = []
    for episode_id in range(int(episodes)):
        shape = shape_list[episode_id % len(shape_list)]
        schedule.append(EpisodeSpec(episode_id, shape, seen[shape], int(seed) + episode_id))
        seen[shape] += 1
    return schedule


def total_episode_count(
    shape_count: int,
    *,
    episodes: int | None,
    episodes_per_shape: int | None,
) -> int:
    """Resolve the new total budget or the backwards-compatible legacy one."""
    if episodes is not None and episodes_per_shape is not None:
        raise ValueError("pass episodes (total) or episodes_per_shape (legacy), not both")
    if episodes is not None:
        total = int(episodes)
    else:
        per_shape = 10 if episodes_per_shape is None else int(episodes_per_shape)
        total = per_shape * int(shape_count)
    if total < 0:
        raise ValueError("episode count must be non-negative")
    return total


def backend_fields(env: Any) -> dict[str, Any]:
    renderer = getattr(getattr(env, "camera_config", None), "renderer_backend", None)
    if renderer is None:
        renderer = getattr(getattr(env, "renderer", None), "backend_name", None)
    if renderer is None:
        renderer = type(getattr(env, "renderer", None)).__name__
    return {
        "environment_backend": type(env).__name__,
        "renderer_backend": str(renderer),
        "asset_faithful": str(renderer) != "toy_direct",
    }


def initial_state_fields(obs: dict, spec: EpisodeSpec) -> dict[str, Any]:
    pose = np.asarray(obs["pose_error"], dtype=float).reshape(3)
    return {
        "episode_id": spec.episode_id,
        "episode": spec.shape_episode,
        "episode_seed": spec.seed,
        "initial_dx_m": float(pose[0]),
        "initial_dy_m": float(pose[1]),
        "initial_dyaw_deg": float(pose[2]),
    }


def summarize_records(records: Sequence[dict]) -> dict[str, Any]:
    if not records:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "success_rate_ci95_low": 0.0,
            "success_rate_ci95_high": 0.0,
            "success_rate_wilson_95": {"low": 0.0, "high": 0.0},
        }
    successes = sum(bool(record["success"]) for record in records)
    low, high = wilson_interval(successes, len(records))
    summary: dict[str, Any] = {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "success_rate_ci95_low": low,
        "success_rate_ci95_high": high,
        "success_rate_wilson_95": {"low": low, "high": high},
    }
    for field, output in (
        ("steps", "mean_steps"),
        ("final_xy_error_mm", "mean_final_xy_error_mm"),
        ("final_yaw_error_deg", "mean_final_yaw_error_deg"),
        ("inference_latency_ms", "mean_inference_latency_ms"),
        ("control_latency_ms", "mean_control_latency_ms"),
    ):
        values = [float(record[field]) for record in records if record.get(field) is not None]
        if values:
            summary[output] = float(np.mean(values))
            summary[f"{field}_statistics"] = percentile_summary(values)
    return summary


def grouped_summary(records: Sequence[dict], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(str(record.get(key, "unknown")), []).append(record)
    return {name: summarize_records(group) for name, group in sorted(groups.items())}
