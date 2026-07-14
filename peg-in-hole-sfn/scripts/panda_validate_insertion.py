from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfn.config import EnvironmentConfig, InsertionConfig
from sfn.geometry import physical_to_normalized_action
from sfn.panda import PandaConfig, PandaPegInHoleInsertionEnv


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate measured incremental Panda insertion")
    ap.add_argument("--shape", default="square-concave1")
    ap.add_argument("--misaligned", action="store_true")
    ap.add_argument("--mode", choices=("kinematic", "dynamic"), default="kinematic")
    ap.add_argument("--increment-mm", type=float, default=0.25)
    ap.add_argument("--target-depth-mm", type=float, default=8.0)
    ap.add_argument("--max-attempts", type=int, default=64)
    ap.add_argument("--out", default="artifacts/panda_validation/insertion_smoke.json")
    ap.add_argument("--peg-mass-kg", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.5)
    ap.add_argument("--max-force", type=float, default=600.0)
    args = ap.parse_args()

    pose = [0.002, 0.0, 0.0] if args.misaligned else [0.0, 0.0, 0.0]
    env = PandaPegInHoleInsertionEnv(
        shapes=[args.shape],
        env_config=EnvironmentConfig(xy_success_axis_mm=3.0, yaw_success_deg=2.0),
        panda_config=PandaConfig(
            execution_mode=args.mode,
            peg_mass_kg=args.peg_mass_kg,
            peg_lateral_friction=args.friction,
            fixture_lateral_friction=args.friction,
            max_force=args.max_force,
        ),
        insertion_config=InsertionConfig(
            descent_increment_mm=args.increment_mm,
            target_depth_mm=args.target_depth_mm,
            max_descent_attempts=args.max_attempts,
        ),
    )
    try:
        obs, _ = env.reset(seed=1, options={"shape": args.shape, "pose_error": pose, "nontrivial": False})
        # Exact smoke uses the oracle no-op at an aligned pose.  The negative
        # case deliberately preserves its 2 mm offset; correcting it here used
        # to make ``--misaligned`` an accidental success test.
        action = (
            [0.0, 0.0, 0.0]
            if args.misaligned
            else physical_to_normalized_action(
                -obs["pose_error"], env.config.max_action_xy_mm, env.config.max_action_yaw_deg
            )
        )
        _, reward, terminated, truncated, info = env.step(action)
        report = {
            "shape": args.shape,
            "execution_mode": args.mode,
            "success": bool(info["insertion_success"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reason": info["termination_reason"],
            "attempts": int(info["insertion_attempts"]),
            "measured_tip_depth_mm": float(info["insertion_depth_mm"]),
            "lateral_drift_mm": float(info["lateral_drift_mm"]),
            "contact_samples": len(info["insertion_contacts"]),
            "max_contact_force": float(info["max_contact_force"]),
            "reward": float(reward),
        }
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
