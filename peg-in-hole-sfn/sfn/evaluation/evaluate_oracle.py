"""Oracle evaluation using the same episode and trace contract as learned methods."""

from __future__ import annotations

import time

import numpy as np

from ..config import CameraConfig, EnvironmentConfig, InsertionConfig
from ..envs import PegInHoleAlignmentEnv, PegInHoleInsertionEnv
from ..models.controllers import OracleController
from .evaluate_contract import backend_fields, episode_schedule, initial_state_fields


def evaluate_oracle(
    *,
    shapes: list[str] | None,
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
    controller = OracleController(env.config.max_action_xy_mm, env.config.max_action_yaw_deg)
    records: list[dict] = []
    steps: list[dict] = []
    try:
        backend = backend_fields(env)
        for spec in episode_schedule(env.shapes, episodes, seed):
            obs, info = env.reset(seed=spec.seed, options={"shape": spec.shape})
            initial = initial_state_fields(obs, spec)
            total_reward = 0.0
            terminated = truncated = False
            local_step = 0
            inference_latencies = []
            control_latencies = []
            while not (terminated or truncated):
                inference_started = time.perf_counter()
                action = controller.act_from_pose_error(obs["pose_error"])
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                control_started = time.perf_counter()
                next_obs, reward, terminated, truncated, next_info = env.step(action.normalized)
                control_ms = (time.perf_counter() - control_started) * 1000.0
                inference_latencies.append(inference_ms)
                control_latencies.append(control_ms)
                total_reward += float(reward)
                steps.append(
                    {
                        **backend,
                        "method": "oracle",
                        "shape": spec.shape,
                        "episode_id": spec.episode_id,
                        "episode": spec.shape_episode,
                        "episode_seed": spec.seed,
                        "step": local_step,
                        "task": task,
                        "mask_source": "oracle_pose_error",
                        "xy_error_mm": float(info["xy_error_mm"]),
                        "yaw_error_deg": float(info["yaw_error_deg"]),
                        "action_x": float(action.normalized[0]),
                        "action_y": float(action.normalized[1]),
                        "action_yaw": float(action.normalized[2]),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success_after_step": bool(next_info["success"]),
                        "inference_ms": inference_ms,
                        "control_ms": control_ms,
                    }
                )
                obs, info = next_obs, next_info
                local_step += 1
            records.append(
                {
                    **backend,
                    **initial,
                    "shape": spec.shape,
                    "task": task,
                    "method": "oracle",
                    "mask_source": "oracle_pose_error",
                    "success": bool(info["success"]),
                    "steps": int(info["step"]),
                    "reward": total_reward,
                    "final_xy_error_mm": float(info["xy_error_mm"]),
                    "final_yaw_error_deg": float(info["yaw_error_deg"]),
                    "termination_reason": info.get(
                        "termination_reason", "success" if info.get("success") else "truncated"
                    ),
                    "insertion_depth_mm": info.get("insertion_depth_mm"),
                    "collision_failure": info.get("collision_failure"),
                    "inference_latency_ms": float(np.mean(inference_latencies)),
                    "control_latency_ms": float(np.mean(control_latencies)),
                }
            )
    finally:
        env.close()
    return records, steps
