import argparse
import math
import random

import gym
import gymEnv  # noqa: F401 - registers the local gym environment


def run_episode(env, episode, gain_xy, gain_yaw, tolerance_xy, tolerance_yaw, max_steps):
    obs = env.reset()

    init_dx = random.uniform(-0.01, 0.01)
    init_dy = random.uniform(-0.01, 0.01)
    init_dyaw = random.uniform(-10.0, 10.0)
    obs, _, _, _ = env.step([init_dx, init_dy, init_dyaw])

    print(
        f"episode {episode}: initial "
        f"dx={obs['dxy'][0] * 1000:.2f}mm, "
        f"dy={obs['dxy'][1] * 1000:.2f}mm, "
        f"yaw={obs['dyaw']:.2f}deg"
    )

    for step in range(max_steps):
        dx, dy = obs["dxy"]
        dyaw = obs["dyaw"]

        xy_error = math.hypot(dx, dy)
        if xy_error <= tolerance_xy and abs(dyaw) <= tolerance_yaw:
            print(
                f"  success at step {step}: "
                f"xy_error={xy_error * 1000:.3f}mm, yaw_error={dyaw:.3f}deg"
            )
            return True

        action = [-gain_xy * dx, -gain_xy * dy, -gain_yaw * dyaw]
        obs, _, _, _ = env.step(action)

        print(
            f"  step {step + 1:02d}: action=({action[0] * 1000:.2f}mm, "
            f"{action[1] * 1000:.2f}mm, {action[2]:.2f}deg), "
            f"remaining=({obs['dxy'][0] * 1000:.2f}mm, "
            f"{obs['dxy'][1] * 1000:.2f}mm, {obs['dyaw']:.2f}deg)"
        )

    dx, dy = obs["dxy"]
    print(
        f"  failed: final xy_error={math.hypot(dx, dy) * 1000:.3f}mm, "
        f"yaw_error={obs['dyaw']:.3f}deg"
    )
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peg_type", default="square-concave1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--gain_xy", type=float, default=0.7)
    parser.add_argument("--gain_yaw", type=float, default=0.7)
    parser.add_argument("--tolerance_xy_mm", type=float, default=1.0)
    parser.add_argument("--tolerance_yaw_deg", type=float, default=2.0)
    args = parser.parse_args()

    random.seed(args.seed)

    env = gym.make(
        "gymEnv:peg-in-hole-v11",
        peg_type=args.peg_type,
        seed=args.seed,
        test_mode=True,
        disable_env_checker=True,
    )

    successes = 0
    try:
        for episode in range(args.episodes):
            ok = run_episode(
                env,
                episode,
                gain_xy=args.gain_xy,
                gain_yaw=args.gain_yaw,
                tolerance_xy=args.tolerance_xy_mm / 1000.0,
                tolerance_yaw=args.tolerance_yaw_deg,
                max_steps=args.max_steps,
            )
            successes += int(ok)
    finally:
        env.close()

    print(f"successes: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
