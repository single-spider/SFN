from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path

import numpy as np
from sfn.config import CameraConfig
from sfn.evaluation.artifacts import write_records_csv
from sfn.evaluation.disturbance import ROBUSTNESS_PROFILES, EnsembleVirtualSensorNetwork, disturb_observation
from sfn.evaluation.evaluate_mfms import load_mfms_policy
from sfn.evaluation.evaluate_sfms import _obs_to_state, load_sfms_policy
from sfn.evaluation.provenance import write_run_provenance
from sfn.geometry import physical_to_normalized_action
from sfn.models.controllers import SFSSController
from sfn.models.vsn import VirtualSensorNetwork
from sfn.panda.artifacts import write_panda_per_shape
from sfn.panda.config import PandaConfig
from sfn.panda.native_vsn import PandaBodyIdGeometricVSN, PandaTopdownTemplateVSN
from sfn.panda.panda_alignment_env import PandaPegInHoleAlignmentEnv
from sfn.panda.panda_insertion_env import PandaPegInHoleInsertionEnv
from sfn.panda.validation import summarize_episodes, write_records
from sfn.training.train_mfms import make_mfms_history_state
from sfn.training.train_sfms import _require_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sfss", "sfms", "mfms", "oracle"], default="sfms")
    ap.add_argument("--task", choices=["alignment", "insertion"], default="alignment")
    ap.add_argument("--policy", default="models/sfms.pt")
    ap.add_argument("--segmentation", default="models/segmentation.pt")
    ap.add_argument("--position", default="models/position.pt")
    ap.add_argument("--orientation", default="models/orientation.pt")
    ap.add_argument("--mask_source", choices=["ground_truth", "predicted"], default="ground_truth")
    ap.add_argument("--split", default="test_unseen")
    ap.add_argument("--shapes", default=None)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/panda_validation/sfms_alignment_smoke")
    ap.add_argument("--ensemble-samples", type=int, default=1)
    ap.add_argument("--robustness-profile", choices=sorted(ROBUSTNESS_PROFILES), default=None)
    ap.add_argument(
        "--native-camera",
        action="store_true",
        help="Use PyBullet RGB/body-ID mask observations instead of synthetic renderer.",
    )
    ap.add_argument(
        "--camera-ignore-robot-occlusion",
        action="store_true",
        help="Exclude the robot from the controller camera while retaining it in physics.",
    )
    ap.add_argument(
        "--native-geometric-vsn",
        action="store_true",
        help="Use geometric Panda body-ID mask decoder instead of checkpoint VSN.",
    )
    ap.add_argument("--native-template-vsn", action="store_true", help="Use calibrated top-down mesh-template VSN.")
    ap.add_argument("--execution-mode", choices=["kinematic", "dynamic"], default="dynamic")
    args = ap.parse_args()

    torch, _ = _require_torch()
    dev = torch.device(args.device)
    if args.shapes:
        shapes = [s for s in args.shapes.split(",") if s]
    else:
        # Keep explicit-shape Panda validation independent from dataset loading.
        from sfn.data.splits import get_split

        shapes = get_split(args.split)
    policy = load_sfms_policy(args.policy, str(dev)) if args.method == "sfms" else None
    history_len = 0
    if args.method == "mfms":
        policy, history_len = load_mfms_policy(args.policy, str(dev))
    vsn = None
    if args.method != "oracle" and not args.native_template_vsn:
        if args.native_geometric_vsn:
            vsn = PandaBodyIdGeometricVSN()
            args.mask_source = "ground_truth"
        else:
            vsn = VirtualSensorNetwork.from_checkpoints(
                args.segmentation if args.mask_source == "predicted" else None,
                args.position,
                args.orientation,
            )
        if args.ensemble_samples > 1:
            vsn = EnsembleVirtualSensorNetwork(vsn, samples=args.ensemble_samples)
        vsn.to(dev).eval()

    records = []
    steps = []
    panda_cfg = PandaConfig(
        native_camera=args.native_camera,
        camera_ignore_robot_occlusion=args.camera_ignore_robot_occlusion,
        execution_mode=args.execution_mode,
        mesh_derived_alignment_z=True,
        camera_eye_offset_m=(0.0, 0.0, 0.20),
        camera_target_offset_m=(0.0, 0.0, 0.03),
    )
    camera_cfg = CameraConfig(crop_width=500, crop_height=400, fov_y_deg=35.0)
    env_cls = PandaPegInHoleInsertionEnv if args.task == "insertion" else PandaPegInHoleAlignmentEnv
    env = env_cls(shapes=shapes, seed=args.seed, panda_config=panda_cfg, camera_config=camera_cfg)
    out_dir = Path(args.out)
    manifest = write_run_provenance(
        out_dir,
        resolved_config={"panda": asdict(panda_cfg), "camera": asdict(camera_cfg), "environment": asdict(env.config)},
        arguments=vars(args),
        input_paths={
            "policy": args.policy if args.method in {"sfms", "mfms"} else None,
            "segmentation": args.segmentation if args.mask_source == "predicted" else None,
            "position": args.position if not args.native_template_vsn else None,
            "orientation": args.orientation if not args.native_template_vsn else None,
        },
        seed=args.seed,
        backend="panda_native_camera" if args.native_camera else "panda_kinematic_synthetic_obs",
    )
    sfss = (
        SFSSController(
            max_xy_mm=env.config.max_action_xy_mm,
            max_yaw_deg=env.config.max_action_yaw_deg,
            confidence_mode="ignore",
        )
        if args.method == "sfss"
        else None
    )
    try:
        global_episode = 0
        for shape in shapes:
            if args.method != "oracle" and args.native_template_vsn:
                segmentation = None
                if args.mask_source == "predicted":
                    from sfn.evaluation.evaluate_perception import _load_model

                    segmentation = _load_model("segmentation", args.segmentation)
                vsn = (
                    PandaTopdownTemplateVSN(shape, panda_cfg, 500, 400, 35.0, segmentation=segmentation).to(dev).eval()
                )
            for ep in range(args.episodes):
                obs, info = env.reset(seed=args.seed + global_episode, options={"shape": shape})
                episode_seed = args.seed + global_episode
                initial_pose = np.asarray(obs["pose_error"], dtype=float).copy()
                if vsn is not None and hasattr(vsn, "reset_state"):
                    vsn.reset_state()
                if sfss is not None:
                    sfss.reset()
                history = []
                total = 0.0
                episode_step_start = len(steps)
                terminated = truncated = False
                while not (terminated or truncated):
                    before_pose = np.asarray(obs["pose_error"], dtype=float).copy()
                    controller_obs = (
                        disturb_observation(
                            obs,
                            ROBUSTNESS_PROFILES[args.robustness_profile],
                            episode_seed=episode_seed,
                            frame_index=int(info["step"]),
                        )
                        if args.robustness_profile is not None
                        else obs
                    )
                    inference_started = time.perf_counter()
                    if args.method == "oracle":
                        pose = obs["pose_error"]
                        action = physical_to_normalized_action(
                            [-pose[0], -pose[1], -pose[2]], env.config.max_action_xy_mm, env.config.max_action_yaw_deg
                        )
                    elif args.method == "sfss":
                        with torch.no_grad():
                            if args.mask_source == "predicted":
                                rgb = torch.as_tensor(controller_obs["rgb"][None], dtype=torch.float32, device=dev)
                                out_vsn = vsn(rgb=rgb)
                            else:
                                mask = torch.as_tensor(controller_obs["mask"][None], dtype=torch.long, device=dev)
                                out_vsn = vsn(mask=mask)
                            action = sfss.act(out_vsn).normalized
                    elif args.method == "sfms":
                        with torch.no_grad():
                            state = _obs_to_state(controller_obs, vsn, args.mask_source, str(dev))
                            if float(state.abs().sum()) == 0.0:
                                action = np.zeros(3, dtype=np.float32)
                            else:
                                mean, _ = policy(state)
                                action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
                    else:
                        with torch.no_grad():
                            state = _obs_to_state(controller_obs, vsn, args.mask_source, str(dev))
                            history.append(state)
                            seq = make_mfms_history_state(history, history_len, str(dev))
                            if float(seq.abs().sum()) == 0.0:
                                action = np.zeros(3, dtype=np.float32)
                            else:
                                mean, _value, _hidden = policy(seq)
                                action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
                    inference_latency_ms = (time.perf_counter() - inference_started) * 1000.0
                    control_started = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(action)
                    control_latency_ms = (time.perf_counter() - control_started) * 1000.0
                    total += float(reward)
                    steps.append(
                        {
                            "shape": shape,
                            "episode": ep,
                            "episode_id": global_episode,
                            "episode_seed": episode_seed,
                            "step": int(info["step"]),
                            "method": args.method,
                            "task": args.task,
                            "mask_source": args.mask_source,
                            "pre_dx_mm": float(before_pose[0] * 1000.0),
                            "pre_dy_mm": float(before_pose[1] * 1000.0),
                            "pre_dyaw_deg": float(before_pose[2]),
                            "action_x": float(action[0]),
                            "action_y": float(action[1]),
                            "action_yaw": float(action[2]),
                            "commanded_dx_m": None if info["action_physical"] is None else float(info["action_physical"][0]),
                            "commanded_dy_m": None if info["action_physical"] is None else float(info["action_physical"][1]),
                            "commanded_dyaw_deg": None if info["action_physical"] is None else float(info["action_physical"][2]),
                            "measured_dx_m": None if info["measured_action_physical"] is None else float(info["measured_action_physical"][0]),
                            "measured_dy_m": None if info["measured_action_physical"] is None else float(info["measured_action_physical"][1]),
                            "measured_dyaw_deg": None if info["measured_action_physical"] is None else float(info["measured_action_physical"][2]),
                            "joint_positions": np.asarray(info["joint_positions"]).tolist(),
                            "joint_target": None
                            if info["joint_target"] is None
                            else np.asarray(info["joint_target"]).tolist(),
                            "joint_limit_margins": np.asarray(info["joint_limit_margins"]).tolist(),
                            "min_joint_limit_margin": info["min_joint_limit_margin"],
                            "joint_limit_violation": info["joint_limit_violation"],
                            "ik_residual_m": info["ik_residual_m"],
                            "ik_branch": info["ik_branch"],
                            "contact_count": info["contact_count"],
                            "contacts": info["contacts"],
                            "max_contact_force": info["max_contact_force"],
                            "max_penetration_mm": info["max_penetration_mm"],
                            "xy_error_mm": float(info["xy_error_mm"]),
                            "yaw_error_deg": float(info["yaw_error_deg"]),
                            "tracking_error_mm": float(info["tracking_error_mm"]),
                            "tracking_yaw_error_deg": float(info["tracking_yaw_error_deg"]),
                            "insertion_depth_mm": info.get("insertion_depth_mm"),
                            "insertion_trace": info.get("insertion_trace"),
                            "lateral_drift_mm": info.get("lateral_drift_mm"),
                            "failure_state": info.get("failure_state"),
                            "collision_failure": info.get("collision_failure"),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "inference_latency_ms": inference_latency_ms,
                            "control_latency_ms": control_latency_ms,
                            "robustness_profile": args.robustness_profile,
                        }
                    )
                succeeded = bool(info.get("insertion_success")) if args.task == "insertion" else bool(info["success"])
                episode_steps = steps[episode_step_start:]
                records.append(
                    {
                        "shape": shape,
                        "episode": ep,
                        "episode_id": global_episode,
                        "episode_seed": episode_seed,
                        "initial_dx_m": float(initial_pose[0]),
                        "initial_dy_m": float(initial_pose[1]),
                        "initial_dyaw_deg": float(initial_pose[2]),
                        "method": args.method,
                        "task": args.task,
                        "mask_source": args.mask_source,
                        "success": succeeded,
                        "alignment_success": bool(info["success"]),
                        "steps": int(info["step"]),
                        "reward": float(total),
                        "inference_latency_ms": float(np.mean([row["inference_latency_ms"] for row in episode_steps])),
                        "control_latency_ms": float(np.mean([row["control_latency_ms"] for row in episode_steps])),
                        "final_xy_error_mm": float(info["xy_error_mm"]),
                        "final_yaw_error_deg": float(info["yaw_error_deg"]),
                        "tracking_error_mm": float(info["tracking_error_mm"]),
                        "tracking_yaw_error_deg": float(info["tracking_yaw_error_deg"]),
                        "insertion_depth_mm": info.get("insertion_depth_mm"),
                        "collision_failure": info.get("collision_failure"),
                        "lateral_drift_mm": info.get("lateral_drift_mm"),
                        "contact_count": info.get("contact_count"),
                        "max_contact_force": info.get("max_contact_force"),
                        "max_penetration_mm": info.get("max_penetration_mm"),
                        "min_joint_limit_margin": info.get("min_joint_limit_margin"),
                        "termination_reason": info.get("termination_reason"),
                        "failure_category": None if succeeded else info.get("termination_reason", "unknown"),
                        "failure_state": None if succeeded else info.get("failure_state", info.get("termination_reason")),
                        "native_camera": bool(args.native_camera),
                        "native_geometric_vsn": bool(args.native_geometric_vsn),
                        "native_template_vsn": bool(args.native_template_vsn),
                        "execution_mode": args.execution_mode,
                        "ensemble_samples": int(args.ensemble_samples),
                        "history_len": history_len if args.method == "mfms" else None,
                        "robustness_profile": args.robustness_profile,
                        "resolved_config_sha256": manifest["resolved_config_sha256"],
                        "policy_sha256": (manifest["inputs"].get("policy") or {}).get("sha256"),
                        "segmentation_sha256": (manifest["inputs"].get("segmentation") or {}).get("sha256"),
                    }
                )
                global_episode += 1
    finally:
        env.close()

    summary = summarize_episodes(records)
    summary["native_camera"] = bool(args.native_camera)
    summary["native_geometric_vsn"] = bool(args.native_geometric_vsn)
    summary["native_template_vsn"] = bool(args.native_template_vsn)
    summary["execution_mode"] = args.execution_mode
    summary["task"] = args.task
    summary["ensemble_samples"] = int(args.ensemble_samples)
    summary["robustness_profile"] = args.robustness_profile
    write_records(out_dir, records, summary, filename="episodes.csv")
    write_records_csv(out_dir / "steps.csv", steps)
    write_panda_per_shape(out_dir / "per_shape.csv", records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
