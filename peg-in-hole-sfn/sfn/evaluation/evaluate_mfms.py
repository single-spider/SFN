"""MFMS recurrent policy evaluation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import CameraConfig, EnvironmentConfig, InsertionConfig
from ..envs import PegInHoleAlignmentEnv, PegInHoleInsertionEnv
from ..models.vsn import VirtualSensorNetwork
from ..training.common import assert_checkpoint_compatible, file_sha256, load_checkpoint_cpu
from ..training.train_mfms import MFMSActorCritic, _obs_to_state, make_mfms_history_state
from ..training.train_sfms import _require_torch
from .evaluate_contract import backend_fields, episode_schedule, initial_state_fields, total_episode_count


def load_mfms_policy(
    checkpoint_path: str | Path,
    device: str = "cpu",
    *,
    expected_compatibility: dict | None = None,
    allow_incompatible: bool = False,
) -> tuple[MFMSActorCritic, int]:
    torch, _ = _require_torch()
    ckpt = load_checkpoint_cpu(checkpoint_path)
    if expected_compatibility is not None:
        assert_checkpoint_compatible(ckpt, expected_compatibility, allow_incompatible=allow_incompatible)
    cfg = ckpt.get("model_config", {})
    model = MFMSActorCritic(
        input_dim=cfg.get("input_dim", 452),
        projection_dim=cfg.get("projection_dim", 256),
        hidden_dim=cfg.get("hidden_dim", 256),
        action_dim=cfg.get("action_dim", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(torch.device(device)).eval()
    return model, int(cfg.get("history_len", 4))


def evaluate_mfms(
    policy_path: str | Path,
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
    shapes: list[str] | None = None,
    episodes_per_shape: int | None = None,
    episodes: int | None = None,
    seed: int = 1,
    mask_source: str = "ground_truth",
    task: str = "alignment",
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    insertion_config: InsertionConfig | None = None,
    vsn: VirtualSensorNetwork | None = None,
    device: str = "cpu",
    return_steps: bool = False,
    enforce_compatibility: bool = False,
    allow_incompatible: bool = False,
) -> list[dict] | tuple[list[dict], list[dict]]:
    """Evaluate deterministic actor-mean MFMS with reset recurrent history."""
    torch, _ = _require_torch()
    dev = torch.device(device if device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    expected = {
        "renderer_backend": getattr(camera_config or CameraConfig(), "renderer_backend", None),
        "mask_source": mask_source,
        "segmentation_sha256": file_sha256(segmentation_path) if mask_source == "predicted" else None,
        "position_sha256": file_sha256(position_path),
        "orientation_sha256": file_sha256(orientation_path),
    }
    policy, history_len = load_mfms_policy(
        policy_path,
        str(dev),
        expected_compatibility=expected if enforce_compatibility else None,
        allow_incompatible=allow_incompatible,
    )
    if vsn is None:
        vsn = VirtualSensorNetwork.from_checkpoints(
            segmentation_path if mask_source == "predicted" else None,
            position_path,
            orientation_path,
        )
    vsn.to(dev).eval()
    if task not in {"alignment", "insertion"}:
        raise ValueError("task must be alignment or insertion")
    env_cls = PegInHoleInsertionEnv if task == "insertion" else PegInHoleAlignmentEnv
    env_kwargs = {
        "shapes": shapes or ["synthetic-square"],
        "seed": seed,
        "env_config": env_config,
        "camera_config": camera_config,
    }
    if task == "insertion":
        env_kwargs["insertion_config"] = insertion_config
    env = env_cls(**env_kwargs)
    records: list[dict] = []
    step_records: list[dict] = []
    try:
        total_episodes = total_episode_count(len(env.shapes), episodes=episodes, episodes_per_shape=episodes_per_shape)
        backend = backend_fields(env)
        for spec in episode_schedule(env.shapes, total_episodes, seed):
            shape, ep = spec.shape, spec.shape_episode
            obs, info = env.reset(seed=spec.seed, options={"shape": shape})
            initial = initial_state_fields(obs, spec)
            set_episode_seed = getattr(vsn, "set_episode_seed", None)
            if callable(set_episode_seed):
                set_episode_seed(spec.seed)
            reset_vsn = getattr(vsn, "reset_state", None)
            if callable(reset_vsn):
                reset_vsn()
            history = []
            terminated = truncated = False
            total = 0.0
            local_step = 0
            inference_latencies = []
            control_latencies = []
            while not (terminated or truncated):
                inference_started = time.perf_counter()
                state, _out_vsn = _obs_to_state(obs, vsn, mask_source, str(dev))
                history.append(state)
                seq = make_mfms_history_state(history, history_len, str(dev))
                with torch.no_grad():
                    if float(seq.abs().sum()) == 0.0:
                        action = np.zeros(3, dtype=np.float32)
                    else:
                        mean, _value, _hidden = policy(seq)
                        action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                control_started = time.perf_counter()
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                control_ms = (time.perf_counter() - control_started) * 1000.0
                inference_latencies.append(inference_ms)
                control_latencies.append(control_ms)
                total += float(reward)
                step_records.append(
                    {
                        **backend,
                        "method": "mfms",
                        "shape": shape,
                        "episode_id": spec.episode_id,
                        "episode": ep,
                        "episode_seed": spec.seed,
                        "step": local_step,
                        "task": task,
                        "mask_source": mask_source,
                        "history_size": min(len(history), history_len),
                        "xy_error_mm": float(info["xy_error_mm"]),
                        "yaw_error_deg": float(info["yaw_error_deg"]),
                        "action_x": float(action[0]),
                        "action_y": float(action[1]),
                        "action_yaw": float(action[2]),
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
                    "shape": shape,
                    "episode": ep,
                    "task": task,
                    "method": "mfms",
                    "mask_source": mask_source,
                    "success": bool(info["success"]),
                    "steps": int(info["step"]),
                    "reward": float(total),
                    "final_xy_error_mm": float(info["xy_error_mm"]),
                    "final_yaw_error_deg": float(info["yaw_error_deg"]),
                    "termination_reason": info.get(
                        "termination_reason", "success" if info.get("success") else "truncated"
                    ),
                    "history_len": history_len,
                    "insertion_depth_mm": info.get("insertion_depth_mm"),
                    "collision_failure": info.get("collision_failure"),
                    "inference_latency_ms": float(np.mean(inference_latencies)),
                    "control_latency_ms": float(np.mean(control_latencies)),
                }
            )
    finally:
        env.close()
    return (records, step_records) if return_steps else records
