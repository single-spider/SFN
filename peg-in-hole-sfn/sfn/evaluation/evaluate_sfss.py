"""SFSS closed-loop evaluation.

This module evaluates the supervised single-frame controller without exposing
``obs["pose_error"]`` to the learned controller path.  Pose error is recorded
only for metrics and the separate oracle baseline.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CameraConfig, EnvironmentConfig, InsertionConfig
from ..envs import PegInHoleAlignmentEnv, PegInHoleInsertionEnv
from ..models import SFSSController, VirtualSensorNetwork
from .evaluate_contract import (
    backend_fields,
    episode_schedule,
    grouped_summary,
    initial_state_fields,
    summarize_records,
    total_episode_count,
)
from .visuals import mask_to_rgb, overlay_mask


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for SFSS evaluation.") from exc
    return torch


def _make_env(
    task: str,
    shapes,
    seed: int,
    env_config: EnvironmentConfig | None,
    insertion_config: InsertionConfig | None,
    camera_config: CameraConfig | None = None,
):
    if task == "insertion":
        return PegInHoleInsertionEnv(
            shapes=shapes,
            seed=seed,
            env_config=env_config,
            insertion_config=insertion_config,
            camera_config=camera_config,
        )
    return PegInHoleAlignmentEnv(shapes=shapes, seed=seed, env_config=env_config, camera_config=camera_config)


def _frame_panel(obs: dict, info: dict, pred_mask: np.ndarray | None, text: str):
    from PIL import Image, ImageDraw

    rgb = np.transpose(obs["rgb"], (1, 2, 0)).astype(np.uint8)
    gt_mask = obs["mask"].astype(np.uint8)
    panels = [
        ("RGB", Image.fromarray(rgb, "RGB")),
        ("GT mask", Image.fromarray(mask_to_rgb(gt_mask), "RGB")),
        ("GT overlay", Image.fromarray(overlay_mask(rgb, gt_mask), "RGB")),
    ]
    if pred_mask is not None:
        panels.extend(
            [
                ("Pred mask", Image.fromarray(mask_to_rgb(pred_mask), "RGB")),
                ("Pred overlay", Image.fromarray(overlay_mask(rgb, pred_mask), "RGB")),
            ]
        )
    for label, img in panels:
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
        d.text((4, 3), label, fill=(255, 255, 255))
    w, h = panels[0][1].size
    canvas = Image.new("RGB", (len(panels) * w, h + 38), (20, 20, 20))
    for i, (_, img) in enumerate(panels):
        canvas.paste(img, (i * w, 0))
    d = ImageDraw.Draw(canvas)
    d.text((4, h + 6), text, fill=(255, 255, 255))
    return canvas


def evaluate_sfss(
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
    shapes: list[str] | None = None,
    episodes_per_shape: int | None = None,
    episodes: int | None = None,
    seed: int = 1,
    task: str = "alignment",
    mask_source: str = "ground_truth",
    recursive: bool = True,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    insertion_config: InsertionConfig | None = None,
    confidence_mode: str = "scale",
    gain_xy: float = 0.7,
    gain_yaw: float = 0.7,
    save_visuals: bool = False,
    visual_dir: str | Path | None = None,
    vsn: Any | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run SFSS episodes and return ``(episode_records, step_records)``."""
    if mask_source not in {"ground_truth", "predicted"}:
        raise ValueError("mask_source must be ground_truth or predicted")
    torch = _require_torch()

    env = _make_env(task, shapes, seed, env_config, insertion_config, camera_config)
    if vsn is None:
        vsn = VirtualSensorNetwork.from_checkpoints(
            segmentation_path if mask_source == "predicted" else None,
            position_path,
            orientation_path,
        )
    try:
        vsn_device = next(vsn.parameters()).device
    except (AttributeError, StopIteration):
        vsn_device = torch.device("cpu")
    controller = SFSSController(
        gain_xy=gain_xy,
        gain_yaw=gain_yaw,
        max_xy_mm=env.config.max_action_xy_mm,
        max_yaw_deg=env.config.max_action_yaw_deg,
        confidence_mode=confidence_mode,
    )
    visual_root = Path(visual_dir or "artifacts/sfss_visuals")
    if save_visuals:
        visual_root.mkdir(parents=True, exist_ok=True)

    episode_records: list[dict] = []
    step_records: list[dict] = []
    try:
        selected_shapes = list(shapes or env.shapes)
        total = total_episode_count(len(selected_shapes), episodes=episodes, episodes_per_shape=episodes_per_shape)
        for spec in episode_schedule(selected_shapes, total, seed):
            shape, ep = spec.shape, spec.shape_episode
            obs, info = env.reset(seed=spec.seed, options={"shape": shape})
            initial = initial_state_fields(obs, spec)
            backend = backend_fields(env)
            set_episode_seed = getattr(vsn, "set_episode_seed", None)
            if callable(set_episode_seed):
                set_episode_seed(spec.seed)
            reset_vsn = getattr(vsn, "reset_state", None)
            if callable(reset_vsn):
                reset_vsn()
            controller.reset()
            total_reward = 0.0
            terminated = truncated = False
            local_step = 0
            frames = []
            inference_latencies = []
            control_latencies = []
            while not (terminated or truncated):
                t0 = time.perf_counter()
                with torch.no_grad():
                    if mask_source == "predicted":
                        rgb = torch.as_tensor(obs["rgb"][None], dtype=torch.float32, device=vsn_device)
                        out = vsn(rgb=rgb)
                    else:
                        mask = torch.as_tensor(obs["mask"][None], dtype=torch.long, device=vsn_device)
                        out = vsn(mask=mask)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                action = controller.act(out)
                control_started = time.perf_counter()
                next_obs, reward, terminated, truncated, next_info = env.step(action.normalized)
                control_ms = (time.perf_counter() - control_started) * 1000.0
                inference_latencies.append(infer_ms)
                control_latencies.append(control_ms)
                total_reward += reward
                pred_mask = out.mask[0].detach().cpu().numpy().astype(np.uint8) if mask_source == "predicted" else None
                step_record = {
                    **backend,
                    "method": "sfss_recursive" if recursive else "sfss_one_step",
                    "episode_id": spec.episode_id,
                    "episode_seed": spec.seed,
                    "shape": shape,
                    "episode": ep,
                    "step": local_step,
                    "task": task,
                    "mask_source": mask_source,
                    "recursive": bool(recursive),
                    "xy_error_mm": float(info["xy_error_mm"]),
                    "yaw_error_deg": float(info["yaw_error_deg"]),
                    "pred_dx_m": float(out.dxy_m.detach().cpu()[0, 0]),
                    "pred_dy_m": float(out.dxy_m.detach().cpu()[0, 1]),
                    "pred_dyaw_deg": float(out.dyaw_deg.detach().cpu()[0]),
                    "position_confidence": float(out.position_confidence.detach().cpu()[0]),
                    "orientation_confidence": float(out.orientation_confidence.detach().cpu()[0]),
                    "action_dx_m": float(action.physical[0]),
                    "action_dy_m": float(action.physical[1]),
                    "action_dyaw_deg": float(action.physical[2]),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success_after_step": bool(next_info["success"]),
                    "inference_ms": float(infer_ms),
                    "control_ms": float(control_ms),
                }
                step_records.append(step_record)
                if save_visuals:
                    text = (
                        f"{shape} ep={ep} step={local_step} "
                        f"err=({info['xy_error_mm']:.2f}mm,{info['yaw_error_deg']:.2f}deg) "
                        f"pred=({step_record['pred_dx_m'] * 1000:.1f},{step_record['pred_dy_m'] * 1000:.1f},{step_record['pred_dyaw_deg']:.1f}) "
                        f"act=({step_record['action_dx_m'] * 1000:.1f},{step_record['action_dy_m'] * 1000:.1f},{step_record['action_dyaw_deg']:.1f})"
                    )
                    frames.append(_frame_panel(obs, info, pred_mask, text))
                obs, info = next_obs, next_info
                local_step += 1
                if not recursive:
                    truncated = not terminated
                    if truncated:
                        info["termination_reason"] = "one_step_limit"
                    break
            if save_visuals and frames:
                ep_dir = visual_root / f"{shape}_ep{ep:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                for i, frame in enumerate(frames):
                    frame.save(ep_dir / f"step_{i:03d}.png")
            episode_records.append(
                {
                    **backend,
                    **initial,
                    "shape": shape,
                    "episode": ep,
                    "task": task,
                    "method": "sfss_recursive" if recursive else "sfss_one_step",
                    "mask_source": mask_source,
                    "success": bool(info["success"]),
                    "steps": int(info["step"]),
                    "reward": float(total_reward),
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
    return episode_records, step_records


def summarize_episodes(records: list[dict]) -> dict:
    return summarize_records(records)


def summarize_episodes_by_shape(records: list[dict]) -> dict[str, dict]:
    return grouped_summary(records, "shape")


def write_sfss_outputs(records: list[dict], steps: list[dict], out_dir: str | Path) -> dict:
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_episodes(records)
    if records:
        with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    if steps:
        with (out_dir / "steps.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
            writer.writeheader()
            writer.writerows(steps)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    by_shape = summarize_episodes_by_shape(records)
    (out_dir / "summary_by_shape.json").write_text(json.dumps(by_shape, indent=2) + "\n", encoding="utf-8")
    if by_shape:
        # Keep a spreadsheet-friendly shape table alongside the lossless JSON.
        # Nested distribution dictionaries remain JSON cells rather than being
        # silently discarded from the release artifact.
        keys = sorted({key for values in by_shape.values() for key in values})
        with (out_dir / "per_shape.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["shape", *keys])
            writer.writeheader()
            for shape, values in sorted(by_shape.items()):
                writer.writerow(
                    {
                        "shape": shape,
                        **{
                            key: json.dumps(values.get(key), sort_keys=True)
                            if isinstance(values.get(key), (dict, list))
                            else values.get(key)
                            for key in keys
                        },
                    }
                )
    return summary
