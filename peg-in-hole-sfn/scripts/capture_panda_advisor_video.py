#!/usr/bin/env python
"""Create a slow, close-up Panda insertion evidence video with synchronized segmentation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sfn.config import CameraConfig, InsertionConfig  # noqa: E402
from sfn.evaluation.evaluate_mfms import load_mfms_policy  # noqa: E402
from sfn.evaluation.evaluate_perception import _load_model  # noqa: E402
from sfn.evaluation.evaluate_sfms import load_sfms_policy  # noqa: E402
from sfn.panda import PandaConfig, PandaPegInHoleInsertionEnv  # noqa: E402
from sfn.panda.native_vsn import PandaTopdownTemplateVSN  # noqa: E402
from sfn.training.train_mfms import make_mfms_history_state  # noqa: E402
from sfn.training.train_sfms import make_sfms_state  # noqa: E402


WIDTH, HEIGHT, FPS = 1920, 1080, 30
INK = (234, 240, 237)
MUTED = (154, 172, 164)
GREEN = (77, 206, 142)
BLUE = (64, 132, 235)
ORANGE = (231, 153, 57)
PANEL = (24, 31, 35)
BACKGROUND = (13, 18, 21)


@dataclass
class Keyframe:
    image: Image.Image
    duration_s: float
    record: dict


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def _close_observer_rgb(env: PandaPegInHoleInsertionEnv) -> np.ndarray:
    """Render a fixed close view in which the peg and fixture dominate the frame."""
    scene = env.scene
    assert scene is not None
    p = scene.p
    origin = np.asarray(scene.task_transform.origin_world, dtype=np.float64)
    eye = origin + np.asarray((0.12, -0.20, 0.08))
    target = origin + np.asarray((0.0, 0.0, 0.025))
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0.0, 0.0, 1.0])
    projection = p.computeProjectionMatrixFOV(30.0, 1.40, 0.01, 1.0)
    _w, _h, rgba, _depth, _seg = p.getCameraImage(
        1120,
        800,
        view,
        projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=scene.client_id,
    )
    return np.asarray(rgba, dtype=np.uint8).reshape(800, 1120, 4)[:, :, :3]


def _predicted_mask(vsn, obs: dict, device: str):
    rgb = torch.as_tensor(obs["rgb"][None], dtype=torch.float32, device=device)
    with torch.no_grad():
        output = vsn(rgb=rgb)
    return output, output.mask[0].detach().cpu().numpy().astype(np.uint8)


def _mask_panel(mask: np.ndarray, reference_seam: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[Image.Image, float]:
    """Render the controller's semantic mask and fixed initial seam reference."""
    h, w = mask.shape
    rgb = np.full((h, w, 3), (15, 21, 25), dtype=np.uint8)
    rgb[mask == 2] = ORANGE
    rgb[mask == 1] = BLUE

    contours, _ = cv2.findContours(reference_seam.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (235, 241, 238), 2, cv2.LINE_AA)

    peg = mask == 1
    overlap = np.logical_and(peg, reference_seam).sum()
    union = np.logical_or(peg, reference_seam).sum()
    coverage = 100.0 * float(overlap) / max(1, int(union))

    def centroid(binary: np.ndarray):
        yy, xx = np.nonzero(binary)
        return None if not len(xx) else (int(round(float(xx.mean()))), int(round(float(yy.mean()))))

    peg_center, seam_center = centroid(peg), centroid(reference_seam)
    if peg_center is not None:
        cv2.drawMarker(rgb, peg_center, (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    if peg_center is not None and seam_center is not None:
        cv2.arrowedLine(rgb, peg_center, seam_center, (86, 220, 156), 2, cv2.LINE_AA, tipLength=0.2)

    x0, y0, x1, y1 = roi
    panel = Image.fromarray(rgb[y0:y1, x0:x1], mode="RGB").resize((600, 480), Image.Resampling.NEAREST)
    return panel, coverage


def _fixed_mask_roi(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Choose one fixed close crop so the 2-D map does not pan between steps."""
    foreground = mask > 0
    yy, xx = np.nonzero(foreground)
    if not len(xx):
        return (125, 100, 375, 300)
    cx, cy = float(xx.mean()), float(yy.mean())
    crop_w, crop_h = 220, 176
    x0 = int(np.clip(round(cx - crop_w / 2), 0, mask.shape[1] - crop_w))
    y0 = int(np.clip(round(cy - crop_h / 2), 0, mask.shape[0] - crop_h))
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _compose(
    env: PandaPegInHoleInsertionEnv,
    obs: dict,
    vsn,
    device: str,
    reference_seam: np.ndarray,
    mask_roi: tuple[int, int, int, int],
    *,
    stage: str,
    step: int,
    action: np.ndarray | None,
    depth_mm: float,
    contact_force_n: float,
    note: str,
    shape: str,
    method: str,
    precision_guard: bool,
) -> tuple[Image.Image, dict]:
    state = env.scene.measure()
    output, predicted = _predicted_mask(vsn, obs, device)
    segmentation, coverage = _mask_panel(predicted, reference_seam, mask_roi)
    observer = Image.fromarray(_close_observer_rgb(env), mode="RGB")

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    canvas.paste(observer, (45, 150))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 115), fill=(18, 25, 28))
    draw.rectangle((0, 111, WIDTH, 115), fill=GREEN)
    draw.rounded_rectangle((1245, 150, 1900, 1035), radius=12, fill=PANEL, outline=(60, 75, 70), width=2)
    canvas.paste(segmentation, (1275, 210))
    draw.text((45, 25), "FRANKA PANDA · DYNAMIC PYBULLET", font=_font(23, True), fill=GREEN)
    draw.text((45, 60), "CLOSE-UP VISUAL SERVOING AND PEG INSERTION", font=_font(34, True), fill=INK)
    draw.text(
        (1875, 52),
        f"SHAPE  {shape.removeprefix('square-').upper()}   ·   CONTROLLER  {method.upper()}"
        + (" + INSERTION GUARD" if precision_guard else ""),
        font=_font(20, True), fill=GREEN, anchor="ra",
    )
    draw.text((1290, 165), "SYNCHRONIZED PREDICTED SEGMENTATION", font=_font(20, True), fill=INK)

    draw.rectangle((1290, 715, 1312, 737), fill=BLUE)
    draw.text((1322, 712), "PEG", font=_font(18, True), fill=INK)
    draw.rectangle((1420, 715, 1442, 737), fill=ORANGE)
    draw.text((1452, 712), "VISIBLE SEAM", font=_font(18, True), fill=INK)
    draw.line((1635, 726, 1680, 726), fill=INK, width=3)
    draw.text((1690, 712), "INITIAL SEAM OUTLINE", font=_font(18, True), fill=INK)

    pose = np.asarray(state.pose_error_task, dtype=float)
    xy_mm = float(np.linalg.norm(pose[:2]) * 1000.0)
    yaw_deg = abs(float(pose[2]))
    estimate_xy = float(torch.linalg.vector_norm(output.dxy_m[0]).detach().cpu()) * 1000.0
    estimate_yaw = abs(float(output.dyaw_deg[0].detach().cpu()))
    confidence = float(output.position_confidence[0].detach().cpu())
    action_values = np.zeros(3, dtype=float) if action is None else np.asarray(action, dtype=float)

    metrics = [
        ("PHASE", stage),
        ("CONTROL STEP", f"{step:02d}"),
        ("MEASURED X–Y ERROR", f"{xy_mm:.3f} mm"),
        ("MEASURED YAW ERROR", f"{yaw_deg:.3f}°"),
        ("VSN ESTIMATE", f"{estimate_xy:.3f} mm / {estimate_yaw:.3f}°"),
        ("PEG / SEAM COVERAGE", f"{coverage:.1f}% pixel IoU"),
        ("INSERTION DEPTH", f"{depth_mm:.3f} mm"),
        ("CONTACT FORCE", f"{contact_force_n:.3f} N"),
        ("LAST POLICY ACTION", f"[{action_values[0]:+.3f}, {action_values[1]:+.3f}, {action_values[2]:+.3f}]"),
        ("VSN CONFIDENCE", f"{confidence:.3f}"),
    ]
    y = 765
    for index, (label, value) in enumerate(metrics):
        col = 1290 if index % 2 == 0 else 1590
        row_y = y + (index // 2) * 50
        draw.text((col, row_y), label, font=_font(14, True), fill=MUTED)
        draw.text((col, row_y + 18), value, font=_font(21, True), fill=INK)

    draw.rounded_rectangle((45, 970, 1205, 1035), radius=8, fill=(18, 25, 28))
    draw.text((70, 986), note, font=_font(24, True), fill=INK)
    draw.text((1010, 995), "STATE HELD FOR INSPECTION", font=_font(13, True), fill=GREEN, anchor="ra")

    record = {
        "stage": stage,
        "step": int(step),
        "measured_xy_error_mm": xy_mm,
        "measured_yaw_error_deg": yaw_deg,
        "vsn_xy_magnitude_mm": estimate_xy,
        "vsn_yaw_error_deg": estimate_yaw,
        "segmentation_reference_iou_percent": coverage,
        "insertion_depth_mm": float(depth_mm),
        "contact_force_n": float(contact_force_n),
        "normalized_action": action_values.tolist(),
        "vsn_confidence": confidence,
    }
    return canvas, record


def _write_video(keyframes: list[Keyframe], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer")
    try:
        for keyframe in keyframes:
            bgr = cv2.cvtColor(np.asarray(keyframe.image), cv2.COLOR_RGB2BGR)
            for _ in range(max(1, round(keyframe.duration_s * fps))):
                writer.write(bgr)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("sfms", "mfms"), default="sfms")
    parser.add_argument("--policy", type=Path, default=Path("models/sfms_mesh_v2_rl_best_compatible.pt"))
    parser.add_argument("--segmentation", type=Path, default=Path("models/segmentation_panda_native_topdown_contrast.pt"))
    parser.add_argument("--shape", default="square-concave2")
    parser.add_argument("--seed", type=int, default=9900)
    parser.add_argument("--pose-error-mm", type=float, nargs=2, default=(6.0, -5.0), metavar=("X", "Y"))
    parser.add_argument("--yaw-error-deg", type=float, default=8.0)
    parser.add_argument("--step-hold", type=float, default=1.35)
    parser.add_argument("--insertion-hold", type=float, default=0.32)
    parser.add_argument("--insertion-xy-axis-mm", type=float, default=0.6)
    parser.add_argument("--insertion-yaw-deg", type=float, default=1.0)
    parser.add_argument(
        "--measured-precision-guard", action="store_true",
        help="Use measured pose for the final safety-gated correction before insertion.",
    )
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--out", type=Path, default=Path("artifacts/panda_advisor_video"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.method == "sfms":
        policy = load_sfms_policy(args.policy, device)
        history_len = 1
    else:
        policy, history_len = load_mfms_policy(args.policy, device)
    segmentation = _load_model("segmentation", args.segmentation).to(device).eval()
    panda = PandaConfig(
        execution_mode="dynamic",
        native_camera=True,
        camera_ignore_robot_occlusion=True,
        mesh_derived_alignment_z=True,
        camera_eye_offset_m=(0.0, 0.0, 0.20),
        camera_target_offset_m=(0.0, 0.0, 0.03),
    )
    camera = CameraConfig(crop_width=500, crop_height=400, fov_y_deg=35.0)
    keyframes: list[Keyframe] = []
    history: list[torch.Tensor] = []
    context = {"step": 0, "action": None}
    env_holder: dict[str, PandaPegInHoleInsertionEnv] = {}
    vsn_holder: dict[str, object] = {}
    seam_holder: dict[str, np.ndarray] = {}

    def add_keyframe(obs: dict, stage: str, duration: float, depth=0.0, force=0.0, note="") -> None:
        image, record = _compose(
            env_holder["env"], obs, vsn_holder["vsn"], device, seam_holder["reference"], seam_holder["roi"],
            stage=stage, step=context["step"], action=context["action"], depth_mm=float(depth),
            contact_force_n=float(force), note=note, shape=args.shape, method=args.method,
            precision_guard=args.measured_precision_guard,
        )
        keyframes.append(Keyframe(image, duration, record))

    def insertion_observer(obs: dict, attempt: dict) -> None:
        add_keyframe(
            obs,
            "INSERTION",
            args.insertion_hold,
            attempt.get("insertion_depth_mm", 0.0),
            attempt.get("max_contact_force", 0.0),
            f"Insertion increment {int(attempt['attempt']):02d} completed · measured state",
        )

    env = PandaPegInHoleInsertionEnv(
        shapes=[args.shape], panda_config=panda, camera_config=camera,
        insertion_config=InsertionConfig(
            insertion_xy_axis_mm=args.insertion_xy_axis_mm,
            insertion_yaw_deg=args.insertion_yaw_deg,
        ),
        insertion_observer=insertion_observer,
    )
    env_holder["env"] = env
    try:
        vsn = PandaTopdownTemplateVSN(
            args.shape, panda, camera.crop_width, camera.crop_height, camera.fov_y_deg, segmentation=segmentation,
        ).to(device).eval()
        vsn_holder["vsn"] = vsn
        pose = [args.pose_error_mm[0] / 1000.0, args.pose_error_mm[1] / 1000.0, args.yaw_error_deg]
        obs, _ = env.reset(seed=args.seed, options={"shape": args.shape, "pose_error": pose, "nontrivial": False})
        _, initial_mask = _predicted_mask(vsn, obs, device)
        seam_holder["reference"] = initial_mask == 2
        seam_holder["roi"] = _fixed_mask_roi(initial_mask)
        add_keyframe(obs, "INITIAL STATE", 2.5, note="Initial camera observation · no command issued")

        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            context["step"] += 1
            with torch.no_grad():
                output, _mask = _predicted_mask(vsn, obs, device)
                state = make_sfms_state(output)
                if args.method == "sfms":
                    mean, _value = policy(state)
                else:
                    history.append(state)
                    sequence = make_mfms_history_state(history, history_len, device)
                    mean, _value, _hidden = policy(sequence)
                action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
            guard_active = False
            if args.measured_precision_guard:
                measured = env.scene.measure().pose_error_task
                if np.linalg.norm(measured[:2]) <= 0.0010 and abs(float(measured[2])) <= 2.0:
                    action = np.clip(
                        np.asarray(
                            [
                                -float(measured[0]) * 1000.0 / env.config.max_action_xy_mm,
                                -float(measured[1]) * 1000.0 / env.config.max_action_xy_mm,
                                -float(measured[2]) / env.config.max_action_yaw_deg,
                            ],
                            dtype=np.float32,
                        ),
                        -1.0,
                        1.0,
                    )
                    guard_active = True
            context["action"] = action
            add_keyframe(
                obs, "PRECISION GUARD" if guard_active else "COMMAND", 0.8,
                note=("Measured insertion-safety correction shown before execution" if guard_active else
                      f"Control step {context['step']:02d} · command shown before execution"),
            )
            obs, _reward, terminated, truncated, info = env.step(action)
            insertion_ran = bool(info.get("insertion_attempted")) or int(info.get("insertion_attempts", 0)) > 0
            if not insertion_ran:
                add_keyframe(
                    obs, "ALIGNMENT", args.step_hold,
                    note=f"Control step {context['step']:02d} settled · inspect residual before the next observation",
                )

        final_stage = "SUCCESS" if info.get("insertion_success") else "FAILED"
        add_keyframe(
            obs, final_stage, 3.5, info.get("insertion_depth_mm", 0.0), info.get("max_contact_force", 0.0),
            "Insertion complete · final measured pose held for inspection" if final_stage == "SUCCESS" else
            f"Run stopped · {info.get('termination_reason', 'unknown reason')}",
        )

        args.out.mkdir(parents=True, exist_ok=True)
        stem = f"panda_{args.method}_{args.shape}_closeup_segmented"
        video_path = args.out / f"{stem}.mp4"
        poster_path = args.out / f"{stem}_poster.png"
        manifest_path = args.out / f"{stem}.json"
        _write_video(keyframes, video_path, args.fps)
        keyframes[-1].image.save(poster_path)
        final_state = env.scene.measure()
        summary = {
            "schema": "sfn.panda_advisor_video/v1",
            "video": str(video_path.resolve()),
            "poster": str(poster_path.resolve()),
            "renderer": "PyBullet ER_TINY_RENDERER",
            "segmentation": "controller-predicted semantic mask",
            "method": args.method,
            "measured_precision_guard": bool(args.measured_precision_guard),
            "shape": args.shape,
            "seed": args.seed,
            "initial_pose_error": pose,
            "fps": args.fps,
            "duration_s": sum(frame.duration_s for frame in keyframes),
            "keyframes": len(keyframes),
            "success": bool(info.get("insertion_success")),
            "termination_reason": info.get("termination_reason"),
            "final_xy_error_mm": float(np.linalg.norm(final_state.pose_error_task[:2]) * 1000.0),
            "final_yaw_error_deg": abs(float(final_state.pose_error_task[2])),
            "insertion_depth_mm": float(info.get("insertion_depth_mm", 0.0)),
            "records": [frame.record for frame in keyframes],
            "provenance_note": "All changing images and measurements are captured from one dynamic PyBullet run. Frames are duplicated only to create inspection pauses; robot states are not interpolated.",
        }
        manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
        if not summary["success"] or summary["final_xy_error_mm"] >= 1.0:
            raise SystemExit(2)
    finally:
        env.close()


if __name__ == "__main__":
    main()
