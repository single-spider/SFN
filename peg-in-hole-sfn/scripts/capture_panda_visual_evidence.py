#!/usr/bin/env python
"""Capture deterministic Panda insertion evidence from the native PyBullet camera."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import InsertionConfig
from sfn.panda import PandaConfig, PandaPegInHoleInsertionEnv


@dataclass(frozen=True)
class Scenario:
    name: str
    pose_error: tuple[float, float, float]
    expected_outcome: str


def _native_frame(obs: dict, lines: list[str], *, accent: tuple[int, int, int]) -> Image.Image:
    """Convert one CHW native-camera observation into a legible evidence frame."""
    rgb = np.transpose(np.asarray(obs["rgb"], dtype=np.uint8), (1, 2, 0))
    camera = Image.fromarray(rgb, mode="RGB").resize((750, 600), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (750, 700), (18, 22, 28))
    canvas.paste(camera, (0, 100))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    draw.rectangle((0, 0, 750, 100), fill=(18, 22, 28))
    draw.rectangle((0, 96, 750, 100), fill=accent)
    for index, line in enumerate(lines[:3]):
        draw.text((18, 10 + index * 27), line, fill=(238, 242, 247), font=font)
    draw.text((560, 67), "NATIVE CAMERA", fill=accent, font=font)
    return canvas


def _save_gif(frames: list[Image.Image], path: Path) -> None:
    if len(frames) < 2:
        raise RuntimeError(f"Evidence GIF requires multiple frames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[300] * (len(frames) - 1) + [1200],
        loop=0,
        optimize=False,
    )


def _run_scenario(scenario: Scenario, *, shape: str, seed: int) -> tuple[list[Image.Image], dict]:
    frames: list[Image.Image] = []
    accent = (61, 214, 140) if scenario.expected_outcome == "success" else (255, 103, 103)

    def observe(obs: dict, attempt: dict) -> None:
        frames.append(
            _native_frame(
                obs,
                [
                    f"Panda insertion | {shape} | seed {seed} | dynamic",
                    f"attempt {attempt['attempt']:02d}  depth {attempt['insertion_depth_mm']:.2f} mm",
                    f"contacts {attempt['contact_count']}  max force {attempt['max_contact_force']:.2f} N",
                ],
                accent=accent,
            )
        )

    env = PandaPegInHoleInsertionEnv(
        shapes=[shape],
        panda_config=PandaConfig(execution_mode="dynamic", native_camera=True, command_steps=60),
        insertion_config=InsertionConfig(
            descent_increment_mm=0.5,
            target_depth_mm=3.0,
            max_descent_attempts=12,
            insertion_xy_axis_mm=1.0,
        ),
        insertion_observer=observe,
    )
    try:
        obs, _ = env.reset(
            seed=seed,
            options={"shape": shape, "pose_error": scenario.pose_error, "nontrivial": False},
        )
        frames.append(
            _native_frame(
                obs,
                [
                    f"Panda insertion | {shape} | seed {seed} | dynamic",
                    "attempt 00  measured start pose",
                    "contacts 0  max force 0.00 N",
                ],
                accent=accent,
            )
        )
        obs, _reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0])
        outcome = "success" if info["insertion_success"] else "failure"
        frames.append(
            _native_frame(
                obs,
                [
                    f"Panda insertion | {shape} | seed {seed} | dynamic",
                    f"OUTCOME: {outcome.upper()}  ({info['termination_reason']})",
                    f"depth {info['insertion_depth_mm']:.2f} mm  contacts {len(info['insertion_contacts'])}  "
                    f"max force {info['max_contact_force']:.2f} N",
                ],
                accent=accent,
            )
        )
        if outcome != scenario.expected_outcome:
            raise RuntimeError(
                f"{scenario.name} produced {outcome}, expected {scenario.expected_outcome}: "
                f"{info['termination_reason']}"
            )
        summary = {
            "shape": shape,
            "seed": seed,
            "mode": "dynamic",
            "outcome": outcome,
            "termination_reason": info["termination_reason"],
            "insertion_depth_mm": round(float(info["insertion_depth_mm"]), 6),
            "contact_samples": len(info["insertion_contacts"]),
            "max_contact_force_n": round(float(info["max_contact_force"]), 6),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "native_camera": True,
        }
        return frames, summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/software_completion_20260713/figures"),
    )
    parser.add_argument("--shape", default="square-concave1")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = (
        Scenario("success", (0.0, 0.0, 0.0), "success"),
        Scenario("contact_failure", (0.0008, 0.0, 0.0), "failure"),
    )
    artifacts: list[dict] = []
    for scenario in scenarios:
        frames, summary = _run_scenario(scenario, shape=args.shape, seed=args.seed)
        gif_name = f"panda_native_camera_{scenario.name}.gif"
        _save_gif(frames, args.out_dir / gif_name)
        artifacts.append({"file": gif_name, **summary})
        if scenario.name == "contact_failure":
            screenshot_name = "panda_native_camera_contact_failure.png"
            frames[-1].save(args.out_dir / screenshot_name)
            artifacts.append({"file": screenshot_name, **summary})

    manifest = {
        "schema": "sfn.panda_visual_evidence/v1",
        "renderer": "panda_native_pybullet",
        "artifacts": artifacts,
    }
    manifest_path = args.out_dir / "panda_native_camera_visual_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
