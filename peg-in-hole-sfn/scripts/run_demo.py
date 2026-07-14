#!/usr/bin/env python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse

from sfn.envs import PegInHoleAlignmentEnv
from sfn.models.controllers import OracleController


def main():
    ap = argparse.ArgumentParser(description="Run a text-mode deterministic SFN demo")
    ap.add_argument("--method", default="oracle", choices=["oracle", "sfss"])
    ap.add_argument("--shape", default="square-triangle")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    env = PegInHoleAlignmentEnv(seed=args.seed)
    obs, info = env.reset(seed=args.seed, options={"shape": args.shape})
    ctrl = OracleController(env.config.max_action_xy_mm, env.config.max_action_yaw_deg)
    print("reset", info)
    terminated = truncated = False
    while not (terminated or truncated):
        action = ctrl.act_from_pose_error(obs["pose_error"])
        obs, reward, terminated, truncated, info = env.step(action.normalized)
        print(
            {
                "step": info["step"],
                "reward": reward,
                "pose_error": info["pose_error"].tolist(),
                "success": info["success"],
            }
        )
    env.close()


if __name__ == "__main__":
    main()
