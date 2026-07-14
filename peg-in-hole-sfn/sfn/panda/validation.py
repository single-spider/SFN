"""Reusable Panda validation routines and artifact writers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..config import EnvironmentConfig, InsertionConfig
from ..geometry import physical_to_normalized_action
from .panda_alignment_env import PandaPegInHoleAlignmentEnv
from .panda_insertion_env import PandaPegInHoleInsertionEnv
from .panda_scene import PandaScene


def parse_csv_floats(text: str) -> list[float]:
    return [float(x) for x in str(text).split(",") if str(x).strip()]


def write_records(out_dir: str | Path, records: list[dict], summary: dict, filename: str = "trials.csv") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if records:
        keys = sorted({k for r in records for k in r.keys()})
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "REPORT.md").write_text(
        "# Panda validation report\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8"
    )


def validate_model(shapes: Iterable[str], out_dir: str | Path | None = None):
    records = []
    for shape in shapes:
        try:
            with PandaScene(shape=str(shape)) as scene:
                m = scene.measure()
                meta = scene.metadata()
                records.append(
                    {
                        "shape": str(shape),
                        "valid": True,
                        "num_joints": len(meta["joints"]),
                        "robot_id": meta["robot_id"],
                        "peg_id": meta["peg_id"],
                        "base_id": meta["base_id"],
                        "ee_x": float(m.ee_pos_world[0]),
                        "ee_y": float(m.ee_pos_world[1]),
                        "ee_z": float(m.ee_pos_world[2]),
                        "error": "",
                    }
                )
        except Exception as exc:
            records.append(
                {
                    "shape": str(shape),
                    "valid": False,
                    "num_joints": 0,
                    "robot_id": -1,
                    "peg_id": -1,
                    "base_id": -1,
                    "ee_x": np.nan,
                    "ee_y": np.nan,
                    "ee_z": np.nan,
                    "error": str(exc),
                }
            )
    summary = {
        "shapes": len(records),
        "valid": sum(bool(r["valid"]) for r in records),
        "success": all(bool(r["valid"]) for r in records),
    }
    if out_dir is not None:
        write_records(out_dir, records, summary)
    return records, summary


def validate_attachment(shape: str, steps: int = 1000, out_dir: str | Path | None = None):
    with PandaScene(shape=shape) as scene:
        drift = scene.validate_attachment_drift(steps)
        records = [{"shape": shape, **drift.as_dict()}]
        summary = {"shape": shape, **drift.as_dict(), "success": drift.translation_mm <= 0.05 and drift.yaw_deg <= 0.05}
    if out_dir is not None:
        write_records(out_dir, records, summary)
    return records, summary


def validate_ik_grid(
    shape: str, grid_mm: Iterable[float], grid_yaw_deg: Iterable[float], out_dir: str | Path | None = None
):
    records = []
    with PandaScene(shape=shape) as scene:
        for dx in grid_mm:
            for dy in grid_mm:
                for yaw in grid_yaw_deg:
                    target = np.asarray([float(dx) / 1000.0, float(dy) / 1000.0, float(yaw)])
                    scene.reset_to_pose_error(target)
                    actual = scene.measure().pose_error_task
                    records.append(
                        {
                            "shape": shape,
                            "dx_mm": float(dx),
                            "dy_mm": float(dy),
                            "yaw_deg": float(yaw),
                            "translation_error_mm": float(np.linalg.norm(actual[:2] - target[:2]) * 1000.0),
                            "yaw_error_abs_deg": abs(float(actual[2] - target[2])),
                        }
                    )
    summary = _tracking_summary(records, "translation_error_mm", "yaw_error_abs_deg")
    summary.update(
        {
            "shape": shape,
            "targets": len(records),
            "success": summary["mean_translation_error_mm"] <= 0.30
            and summary["mean_yaw_error_deg"] <= 0.30
            and summary["max_translation_error_mm"] <= 1.0
            and summary["max_yaw_error_deg"] <= 1.0,
        }
    )
    if out_dir is not None:
        write_records(out_dir, records, summary)
    return records, summary


def validate_command_tracking(shape: str, trials: int = 100, out_dir: str | Path | None = None, seed: int = 1):
    rng = np.random.default_rng(seed)
    records = []
    commands = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(max(0, int(trials) - len(commands))):
        commands.append((rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-2, 2)))
    with PandaScene(shape=shape, seed=seed) as scene:
        scene.reset_to_pose_error([0, 0, 0])
        for i, (dx_mm, dy_mm, dyaw) in enumerate(commands):
            before = scene.measure().pose_error_task.copy()
            result = scene.execute_cartesian_delta(float(dx_mm) / 1000.0, float(dy_mm) / 1000.0, float(dyaw))
            measured_delta = scene.measure().pose_error_task - before
            sign_ok = True
            if dx_mm:
                sign_ok = np.sign(measured_delta[0]) == np.sign(dx_mm)
            if dy_mm:
                sign_ok = sign_ok and np.sign(measured_delta[1]) == np.sign(dy_mm)
            if dyaw:
                sign_ok = sign_ok and np.sign(measured_delta[2]) == np.sign(dyaw)
            records.append(
                {
                    "trial": i,
                    "shape": shape,
                    "cmd_dx_mm": float(dx_mm),
                    "cmd_dy_mm": float(dy_mm),
                    "cmd_yaw_deg": float(dyaw),
                    "measured_dx_mm": float(measured_delta[0] * 1000.0),
                    "measured_dy_mm": float(measured_delta[1] * 1000.0),
                    "measured_yaw_deg": float(measured_delta[2]),
                    "tracking_error_mm": float(result.pos_error_mm),
                    "tracking_yaw_error_deg": float(result.yaw_error_deg),
                    "sign_ok": bool(sign_ok),
                }
            )
    summary = _tracking_summary(records, "tracking_error_mm", "tracking_yaw_error_deg")
    summary.update(
        {
            "shape": shape,
            "trials": len(records),
            "cardinal_signs_ok": all(bool(r["sign_ok"]) for r in records[:6]),
            "success": all(bool(r["sign_ok"]) for r in records[:6])
            and summary["mean_translation_error_mm"] <= 0.30
            and summary["mean_yaw_error_deg"] <= 0.30,
        }
    )
    if out_dir is not None:
        write_records(out_dir, records, summary)
    return records, summary


def evaluate_oracle(
    shapes: list[str],
    episodes_per_shape: int = 10,
    seed: int = 1,
    env_config: EnvironmentConfig | None = None,
    out_dir: str | Path | None = None,
):
    env_config = env_config or EnvironmentConfig(xy_success_axis_mm=0.6, yaw_success_deg=1.0)
    records = []
    env = PandaPegInHoleAlignmentEnv(shapes=shapes, env_config=env_config, seed=seed)
    try:
        global_episode = 0
        for shape in shapes:
            for ep in range(episodes_per_shape):
                obs, info = env.reset(seed=seed + global_episode, options={"shape": shape})
                total = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    pose = obs["pose_error"]
                    action_phys = np.asarray([-pose[0], -pose[1], -pose[2]], dtype=np.float32)
                    action = physical_to_normalized_action(
                        action_phys, env.config.max_action_xy_mm, env.config.max_action_yaw_deg
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                records.append(
                    {
                        "shape": shape,
                        "episode": ep,
                        "success": bool(info["success"]),
                        "steps": int(info["step"]),
                        "reward": float(total),
                        "final_xy_error_mm": float(info["xy_error_mm"]),
                        "final_yaw_error_deg": float(info["yaw_error_deg"]),
                        "tracking_error_mm": float(info["tracking_error_mm"]),
                        "tracking_yaw_error_deg": float(info["tracking_yaw_error_deg"]),
                    }
                )
                global_episode += 1
    finally:
        env.close()
    summary = summarize_episodes(records)
    if out_dir is not None:
        write_records(out_dir, records, summary, filename="episodes.csv")
    return records, summary


def evaluate_insertion(shape: str, exact: bool = True, out_dir: str | Path | None = None):
    pose = [0.0, 0.0, 0.0] if exact else [0.002, 0.0, 0.0]
    env = PandaPegInHoleInsertionEnv(
        shapes=[shape],
        env_config=EnvironmentConfig(xy_success_axis_mm=1.0, yaw_success_deg=2.0),
        insertion_config=InsertionConfig(),
    )
    try:
        obs, _ = env.reset(seed=1, options={"shape": shape, "pose_error": pose, "nontrivial": False})
        action = physical_to_normalized_action(
            [-obs["pose_error"][0], -obs["pose_error"][1], -obs["pose_error"][2]],
            env.config.max_action_xy_mm,
            env.config.max_action_yaw_deg,
        )
        _, _, terminated, _, info = env.step(action)
        records = [
            {
                "shape": shape,
                "exact": bool(exact),
                "terminated": bool(terminated),
                "success": bool(info.get("insertion_success", False)),
                "termination_reason": info.get("termination_reason", "unknown"),
                "xy_error_mm": float(info["xy_error_mm"]),
                "yaw_error_deg": float(info["yaw_error_deg"]),
            }
        ]
    finally:
        env.close()
    summary = {
        "shape": shape,
        "exact": bool(exact),
        "success": bool(records[0]["success"]),
        "expected_success": bool(exact),
    }
    if out_dir is not None:
        write_records(out_dir, records, summary)
    return records, summary


def summarize_episodes(records: list[dict]) -> dict:
    if not records:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": 0.0,
            "success_rate_ci95_low": 0.0,
            "success_rate_ci95_high": 0.0,
        }
    from ..evaluation.statistics import percentile_summary, wilson_interval

    successes = sum(bool(r["success"]) for r in records)
    low, high = wilson_interval(successes, len(records))
    summary = {
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "success_rate_ci95_low": low,
        "success_rate_ci95_high": high,
        "success_rate_wilson_95": {"low": low, "high": high},
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "mean_final_xy_error_mm": float(np.mean([r["final_xy_error_mm"] for r in records])),
        "mean_final_yaw_error_deg": float(np.mean([r["final_yaw_error_deg"] for r in records])),
    }
    for field in (
        "steps",
        "final_xy_error_mm",
        "final_yaw_error_deg",
        "insertion_depth_mm",
        "inference_latency_ms",
        "control_latency_ms",
    ):
        values = [float(row[field]) for row in records if row.get(field) is not None]
        if values:
            summary[f"{field}_statistics"] = percentile_summary(values)
    return summary


def _tracking_summary(records: list[dict], trans_key: str, yaw_key: str) -> dict:
    if not records:
        return {
            "mean_translation_error_mm": 0.0,
            "mean_yaw_error_deg": 0.0,
            "max_translation_error_mm": 0.0,
            "max_yaw_error_deg": 0.0,
        }
    return {
        "mean_translation_error_mm": float(np.mean([r[trans_key] for r in records])),
        "mean_yaw_error_deg": float(np.mean([r[yaw_key] for r in records])),
        "max_translation_error_mm": float(np.max([r[trans_key] for r in records])),
        "max_yaw_error_deg": float(np.max([r[yaw_key] for r in records])),
    }
