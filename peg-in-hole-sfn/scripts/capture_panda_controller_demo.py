#!/usr/bin/env python
"""Capture an SFMS predicted-RGB Panda alignment and insertion animation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import CameraConfig  # noqa: E402
from sfn.evaluation.evaluate_mfms import load_mfms_policy  # noqa: E402
from sfn.evaluation.evaluate_perception import _load_model  # noqa: E402
from sfn.evaluation.evaluate_sfms import _obs_to_state, load_sfms_policy  # noqa: E402
from sfn.panda import PandaConfig, PandaPegInHoleInsertionEnv  # noqa: E402
from sfn.panda.native_vsn import PandaTopdownTemplateVSN  # noqa: E402
from sfn.training.train_mfms import make_mfms_history_state  # noqa: E402


def _observer_rgb(env: PandaPegInHoleInsertionEnv) -> np.ndarray:
    scene = env.scene
    p = scene.p
    origin = np.asarray(scene.task_transform.origin_world, dtype=np.float64)
    # Close observer view centred on the grasp, peg and fixture.  The former
    # release camera was far enough away to make attachment and insertion hard
    # to inspect in a browser-sized presentation panel.
    eye = origin + np.asarray((0.16, -0.58, 0.36))
    target = origin + np.asarray((0.00, 0.0, 0.12))
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0.0, 0.0, 1.0])
    projection = p.computeProjectionMatrixFOV(35.0, 4.0 / 3.0, 0.01, 2.0)
    _w, _h, rgba, _depth, _seg = p.getCameraImage(
        800,
        600,
        view,
        projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=scene.client_id,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(600, 800, 4)[:, :, :3]


def _frame(env: PandaPegInHoleInsertionEnv, obs: dict, lines: list[str], *, success: bool = False) -> Image.Image:
    observer = Image.fromarray(_observer_rgb(env), mode="RGB")
    native = Image.fromarray(np.transpose(np.asarray(obs["rgb"], dtype=np.uint8), (1, 2, 0)), mode="RGB")
    native.thumbnail((280, 224), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1120, 650), (18, 22, 28))
    canvas.paste(observer, (0, 50))
    canvas.paste(native, (820, 375))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    accent = (61, 214, 140) if success else (86, 169, 255)
    draw.rectangle((0, 0, 1120, 50), fill=(18, 22, 28))
    draw.rectangle((800, 50, 1120, 650), fill=(18, 22, 28))
    draw.rectangle((0, 46, 1120, 50), fill=accent)
    draw.text((18, 13), "Franka Panda | SFMS | predicted RGB mask | dynamic PyBullet", fill=(242, 245, 249), font=font)
    draw.text((830, 345), "CONTROLLER CAMERA", fill=accent, font=font)
    for index, line in enumerate(lines):
        draw.text((820, 75 + 34 * index), line, fill=(242, 245, 249), font=font)
    return canvas


def _save_gif(frames: list[Image.Image], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[550] * (len(frames) - 1) + [2200],
        optimize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sfms", "mfms"), default="sfms")
    parser.add_argument("--policy", type=Path, default=Path("models/sfms_mesh_v2_rl_best_compatible.pt"))
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=Path("models/segmentation_panda_native_topdown_contrast.pt"),
    )
    parser.add_argument("--shape", default="square-concave2")
    parser.add_argument("--seed", type=int, default=9900)
    parser.add_argument("--out", type=Path, default=Path("artifacts/panda_live_demo_20260714"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = None
    history_len = 1
    if args.method == "sfms":
        policy = load_sfms_policy(args.policy, device)
    else:
        policy, history_len = load_mfms_policy(args.policy, device)
    segmentation = _load_model("segmentation", args.segmentation)
    panda = PandaConfig(
        execution_mode="dynamic",
        native_camera=True,
        camera_ignore_robot_occlusion=True,
        mesh_derived_alignment_z=True,
        camera_eye_offset_m=(0.0, 0.0, 0.20),
        camera_target_offset_m=(0.0, 0.0, 0.03),
    )
    camera = CameraConfig(crop_width=500, crop_height=400, fov_y_deg=35.0)
    env_holder: dict[str, PandaPegInHoleInsertionEnv] = {}
    frames: list[Image.Image] = []

    def insertion_observer(obs: dict, attempt: dict) -> None:
        pose = np.asarray(obs["pose_error"], dtype=float)
        frames.append(
            _frame(
                env_holder["env"],
                obs,
                [
                    "STAGE: INSERTION",
                    f"X-Y error: {np.linalg.norm(pose[:2]) * 1000.0:.3f} mm",
                    f"Yaw error: {abs(pose[2]):.3f} deg",
                    f"Depth: {attempt['insertion_depth_mm']:.2f} mm",
                    f"Contact force: {attempt['max_contact_force']:.2f} N",
                ],
            )
        )

    env = PandaPegInHoleInsertionEnv(
        shapes=[args.shape],
        panda_config=panda,
        camera_config=camera,
        insertion_observer=insertion_observer,
    )
    env_holder["env"] = env
    try:
        vsn = PandaTopdownTemplateVSN(
            args.shape,
            panda,
            camera.crop_width,
            camera.crop_height,
            camera.fov_y_deg,
            segmentation=segmentation,
        ).to(device).eval()
        obs, info = env.reset(seed=args.seed, options={"shape": args.shape})
        initial_pose = np.asarray(obs["pose_error"], dtype=float).copy()
        history: list[torch.Tensor] = []
        frames.append(
            _frame(
                env,
                obs,
                [
                    "STAGE: START",
                    f"X-Y error: {np.linalg.norm(initial_pose[:2]) * 1000.0:.3f} mm",
                    f"Yaw error: {abs(initial_pose[2]):.3f} deg",
                    "Policy input: predicted RGB mask",
                ],
            )
        )
        terminated = truncated = False
        while not (terminated or truncated):
            with torch.no_grad():
                state = _obs_to_state(obs, vsn, "predicted", device)
                if args.method == "sfms":
                    mean, _value = policy(state)
                else:
                    history.append(state)
                    sequence = make_mfms_history_state(history, history_len, device)
                    mean, _value, _hidden = policy(sequence)
                action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
            obs, _reward, terminated, truncated, info = env.step(action)
            frames.append(
                _frame(
                    env,
                    obs,
                    [
                        "STAGE: ALIGNMENT" if not info.get("insertion_attempted") else "STAGE: COMPLETE",
                        f"X-Y error: {float(info['xy_error_mm']):.3f} mm",
                        f"Yaw error: {float(info['yaw_error_deg']):.3f} deg",
                        f"Measured step: {int(info['step'])}",
                    ],
                    success=bool(info.get("insertion_success")),
                )
            )

        summary = {
            "renderer": "panda_observer_plus_native_camera",
            "method": args.method,
            "policy_observation": "predicted_rgb_mask",
            "shape": args.shape,
            "seed": args.seed,
            "execution_mode": "dynamic",
            "success": bool(info.get("insertion_success")),
            "termination_reason": info.get("termination_reason"),
            "initial_xy_error_mm": float(np.linalg.norm(initial_pose[:2]) * 1000.0),
            "initial_yaw_error_deg": float(abs(initial_pose[2])),
            "final_xy_error_mm": float(info["xy_error_mm"]),
            "final_yaw_error_deg": float(info["yaw_error_deg"]),
            "insertion_depth_mm": float(info.get("insertion_depth_mm", 0.0)),
            "steps": int(info["step"]),
            "note": "Pose errors are simulator measurements used only for the visual overlay, not policy input.",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        gif = args.out / f"panda_{args.method}_predicted_submillimetre.gif"
        _save_gif(frames, gif)
        (args.out / f"panda_{args.method}_predicted_submillimetre.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"gif": str(gif.resolve()), **summary}, indent=2))
        if not summary["success"] or summary["final_xy_error_mm"] >= 1.0:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
