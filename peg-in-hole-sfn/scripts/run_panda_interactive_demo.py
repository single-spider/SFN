#!/usr/bin/env python
"""Run an inspectable SFMS Panda insertion in the PyBullet GUI."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import CameraConfig  # noqa: E402
from sfn.evaluation.evaluate_perception import _load_model  # noqa: E402
from sfn.evaluation.evaluate_sfms import _obs_to_state, load_sfms_policy  # noqa: E402
from sfn.panda import PandaConfig, PandaPegInHoleInsertionEnv  # noqa: E402
from sfn.panda.native_vsn import PandaTopdownTemplateVSN  # noqa: E402


class InteractiveCamera:
    """Keyboard-driven debug camera independent of PyBullet mouse bindings."""

    def __init__(self, pybullet_module, client_id: int):
        self.p = pybullet_module
        self.client_id = client_id
        self.distance = 0.18
        self.yaw = 38.0
        self.pitch = -48.0
        self.target = np.asarray([-1.0, 0.0, 0.035], dtype=np.float64)
        self.apply()

    def apply(self) -> None:
        self.p.resetDebugVisualizerCamera(
            cameraDistance=float(self.distance),
            cameraYaw=float(self.yaw),
            cameraPitch=float(self.pitch),
            cameraTargetPosition=self.target.tolist(),
            physicsClientId=self.client_id,
        )

    def poll(self) -> None:
        events = self.p.getKeyboardEvents(physicsClientId=self.client_id)
        active_mask = self.p.KEY_IS_DOWN | self.p.KEY_WAS_TRIGGERED

        def active(key: str) -> bool:
            return bool(events.get(ord(key), 0) & active_mask)

        changed = False
        if active("j"):
            self.yaw -= 2.0
            changed = True
        if active("l"):
            self.yaw += 2.0
            changed = True
        if active("i"):
            self.pitch = min(-2.0, self.pitch + 2.0)
            changed = True
        if active("k"):
            self.pitch = max(-89.0, self.pitch - 2.0)
            changed = True
        if active("w"):
            self.distance = max(0.07, self.distance - 0.008)
            changed = True
        if active("s"):
            self.distance = min(1.5, self.distance + 0.008)
            changed = True
        if active("a"):
            self.target[0] -= 0.004
            changed = True
        if active("d"):
            self.target[0] += 0.004
            changed = True
        if active("u"):
            self.target[1] -= 0.004
            changed = True
        if active("o"):
            self.target[1] += 0.004
            changed = True
        if active("q"):
            self.target[2] += 0.004
            changed = True
        if active("e"):
            self.target[2] -= 0.004
            changed = True
        if active("c"):
            self.distance, self.yaw, self.pitch = 0.18, 38.0, -48.0
            self.target[:] = (-1.0, 0.0, 0.035)
            changed = True
        if changed:
            self.apply()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("models/sfms_mesh_v2_rl_best_compatible.pt"))
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=Path("models/segmentation_panda_native_topdown_contrast.pt"),
    )
    parser.add_argument(
        "--shapes",
        default="square-triangle,square-square,square-hexagon,square-concave2",
        help="Comma-separated shapes cycled on successive replays.",
    )
    parser.add_argument("--seed", type=int, default=9900)
    parser.add_argument("--step-delay", type=float, default=1.2)
    args = parser.parse_args()
    shapes = [value.strip() for value in args.shapes.split(",") if value.strip()]
    if not shapes:
        raise ValueError("--shapes must contain at least one shape")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_sfms_policy(args.policy, device)
    segmentation = _load_model("segmentation", args.segmentation)
    panda = PandaConfig(
        gui=True,
        execution_mode="dynamic",
        native_camera=True,
        # Keep the calibrated top-down eye-to-hand perception feed independent
        # of robot occlusion.  The Panda remains visible in the GUI and fully
        # participates in physics; only the controller camera excludes it.
        camera_ignore_robot_occlusion=True,
        mesh_derived_alignment_z=True,
        camera_eye_offset_m=(0.0, 0.0, 0.20),
        camera_target_offset_m=(0.0, 0.0, 0.03),
    )
    camera = CameraConfig(crop_width=500, crop_height=400, fov_y_deg=35.0)
    env_holder: dict[str, PandaPegInHoleInsertionEnv] = {}
    camera_holder: dict[str, InteractiveCamera] = {}
    debug_ids: list[int] = []

    def interactive_wait(seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            env = env_holder["env"]
            if env.scene is None or not env.scene.p.isConnected(env.scene.client_id):
                return
            camera_holder["camera"].poll()
            time.sleep(0.025)

    def update_status(lines: list[str], color: tuple[float, float, float] = (0.1, 0.8, 1.0)) -> None:
        env = env_holder["env"]
        p, cid = env.scene.p, env.scene.client_id
        for item in debug_ids:
            try:
                p.removeUserDebugItem(item, physicsClientId=cid)
            except Exception:
                pass
        debug_ids.clear()
        for index, line in enumerate(lines):
            debug_ids.append(
                p.addUserDebugText(
                    line,
                    [-1.085, -0.075, 0.145 - index * 0.018],
                    textColorRGB=color,
                    textSize=0.72,
                    lifeTime=0,
                    physicsClientId=cid,
                )
            )

    def insertion_observer(obs: dict, attempt: dict) -> None:
        pose = np.asarray(obs["pose_error"], dtype=float)
        update_status(
            [
                "STAGE: PHYSICAL INSERTION",
                f"X-Y error: {np.linalg.norm(pose[:2]) * 1000.0:.3f} mm",
                f"Yaw error: {abs(pose[2]):.3f} deg",
                f"Insertion depth: {attempt['insertion_depth_mm']:.2f} mm",
                f"Contact force: {attempt['max_contact_force']:.2f} N",
            ],
            (1.0, 0.65, 0.1),
        )
        interactive_wait(args.step_delay)

    env = PandaPegInHoleInsertionEnv(
        shapes=shapes,
        panda_config=panda,
        camera_config=camera,
        insertion_observer=insertion_observer,
    )
    env_holder["env"] = env
    try:
        current_shape = shapes[0]
        obs, info = env.reset(seed=args.seed, options={"shape": current_shape})
        scene = env.scene
        p, cid = scene.p, scene.client_id
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=cid)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)
        camera_holder["camera"] = InteractiveCamera(p, cid)
        run_index = 0
        while p.isConnected(cid):
            if run_index > 0:
                current_shape = shapes[run_index % len(shapes)]
                obs, info = env.reset(seed=args.seed, options={"shape": current_shape})
                scene = env.scene
                p, cid = scene.p, scene.client_id
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=cid)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)
                camera_holder["camera"].p = p
                camera_holder["camera"].client_id = cid
                camera_holder["camera"].apply()
            vsn = PandaTopdownTemplateVSN(
                current_shape,
                panda,
                camera.crop_width,
                camera.crop_height,
                camera.fov_y_deg,
                segmentation=segmentation,
            ).to(device).eval()
            initial = np.asarray(obs["pose_error"], dtype=float)
            update_status(
                [
                    f"{current_shape} | REPLAY {run_index + 1}",
                    f"Initial X-Y: {np.linalg.norm(initial[:2]) * 1000.0:.3f} mm",
                    f"Initial yaw: {abs(initial[2]):.3f} deg",
                    "J/L rotate | I/K tilt | W/S zoom | C reset",
                ]
            )
            interactive_wait(2.5)

            terminated = truncated = False
            while not (terminated or truncated) and p.isConnected(cid):
                camera_holder["camera"].poll()
                with torch.no_grad():
                    state = _obs_to_state(obs, vsn, "predicted", device)
                    mean, _value = policy(state)
                    action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
                obs, _reward, terminated, truncated, info = env.step(action)
                update_status(
                    [
                        "STAGE: ALIGNMENT" if not info.get("insertion_attempted") else "STAGE: COMPLETE",
                        f"Measured X-Y: {float(info['xy_error_mm']):.3f} mm",
                        f"Measured yaw: {float(info['yaw_error_deg']):.3f} deg",
                        f"Controller step: {int(info['step'])}",
                        current_shape,
                    ],
                    (0.1, 1.0, 0.45) if info.get("insertion_success") else (0.1, 0.8, 1.0),
                )
                interactive_wait(args.step_delay)

            if not p.isConnected(cid):
                break
            success = bool(info.get("insertion_success"))
            update_status(
                [
                    "SUCCESS - RESTARTING AUTOMATICALLY" if success else "RUN FINISHED - RESTARTING",
                    f"Final X-Y: {float(info['xy_error_mm']):.3f} mm",
                    f"Final yaw: {float(info['yaw_error_deg']):.3f} deg",
                    f"Depth: {float(info.get('insertion_depth_mm', 0.0)):.3f} mm",
                    f"Next shape: {shapes[(run_index + 1) % len(shapes)]}",
                ],
                (0.1, 1.0, 0.35) if success else (1.0, 0.2, 0.2),
            )
            print(
                f"replay={run_index + 1} success={success} final_xy_mm={float(info['xy_error_mm']):.6f} "
                f"final_yaw_deg={float(info['yaw_error_deg']):.6f} "
                f"depth_mm={float(info.get('insertion_depth_mm', 0.0)):.6f}",
                flush=True,
            )
            run_index += 1
            interactive_wait(4.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
