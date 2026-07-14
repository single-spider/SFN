"""Deterministic random-policy evaluation baseline."""

from __future__ import annotations

import time

import numpy as np

from ..config import CameraConfig, EnvironmentConfig, InsertionConfig
from ..envs import PegInHoleAlignmentEnv, PegInHoleInsertionEnv
from .evaluate_contract import backend_fields, episode_schedule, initial_state_fields


def evaluate_random(
    *,
    shapes: list[str],
    episodes: int,
    seed: int,
    task: str,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    insertion_config: InsertionConfig | None = None,
) -> tuple[list[dict], list[dict]]:
    env_cls = PegInHoleInsertionEnv if task == "insertion" else PegInHoleAlignmentEnv
    kwargs = {"shapes": shapes, "seed": seed, "env_config": env_config, "camera_config": camera_config}
    if task == "insertion":
        kwargs["insertion_config"] = insertion_config
    env = env_cls(**kwargs)
    records, steps = [], []
    try:
        for spec in episode_schedule(shapes, episodes, seed):
            observation, info = env.reset(seed=spec.seed, options={"shape": spec.shape})
            initial = initial_state_fields(observation, spec)
            backend = backend_fields(env)
            rng = np.random.default_rng(spec.seed)
            total_reward = 0.0
            terminated = truncated = False
            inference_latencies = []
            control_latencies = []
            while not (terminated or truncated):
                inference_started = time.perf_counter()
                action = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                before = np.asarray(observation["pose_error"], dtype=float)
                control_started = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                control_ms = (time.perf_counter() - control_started) * 1000.0
                inference_latencies.append(inference_ms)
                control_latencies.append(control_ms)
                total_reward += float(reward)
                steps.append(
                    {
                        **backend,
                        "method": "random",
                        "episode_id": spec.episode_id,
                        "episode_seed": spec.seed,
                        "shape": spec.shape,
                        "episode": spec.shape_episode,
                        "step": int(info["step"]),
                        "task": task,
                        "xy_error_mm": float(np.linalg.norm(before[:2]) * 1000.0),
                        "yaw_error_deg": abs(float(before[2])),
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_yaw": float(action[2]),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "inference_ms": inference_ms,
                        "control_ms": control_ms,
                    }
                )
            success = bool(info.get("insertion_success")) if task == "insertion" else bool(info["success"])
            records.append(
                {
                    **backend,
                    **initial,
                    "shape": spec.shape,
                    "episode": spec.shape_episode,
                    "method": "random",
                    "task": task,
                    "mask_source": "none",
                    "success": success,
                    "steps": int(info["step"]),
                    "reward": total_reward,
                    "final_xy_error_mm": float(info["xy_error_mm"]),
                    "final_yaw_error_deg": float(info["yaw_error_deg"]),
                    "termination_reason": info.get("termination_reason"),
                    "insertion_depth_mm": info.get("insertion_depth_mm"),
                    "inference_latency_ms": float(np.mean(inference_latencies)),
                    "control_latency_ms": float(np.mean(control_latencies)),
                }
            )
    finally:
        env.close()
    return records, steps
