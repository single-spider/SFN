import argparse
import time

import gym
import gymEnv  # noqa: F401 - registers the local gym environment
import pybullet as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peg_type", default="square-concave1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--dx_mm", type=float, default=5.0)
    parser.add_argument("--dy_mm", type=float, default=-4.0)
    parser.add_argument("--dyaw_deg", type=float, default=6.0)
    args = parser.parse_args()

    env = gym.make(
        "gymEnv:peg-in-hole-v11",
        peg_type=args.peg_type,
        seed=args.seed,
        test_mode=True,
        gui_mode=True,
        disable_env_checker=True,
    )

    try:
        env.reset()
        env.step([args.dx_mm / 1000.0, args.dy_mm / 1000.0, args.dyaw_deg])
        print("PyBullet GUI is open.")
        print("Close the PyBullet window or wait for the timer to finish.")

        end_time = time.time() + args.seconds
        while time.time() < end_time:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
