import argparse
import math
import time

import gym
import gymEnv  # noqa: F401 - registers the local gym environment
import numpy as np
import pybullet as p


def sleep_steps(seconds):
    end_time = time.time() + seconds
    while time.time() < end_time:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peg_type", default="square-concave1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dx_mm", type=float, default=8.0)
    parser.add_argument("--dy_mm", type=float, default=-6.0)
    parser.add_argument("--dyaw_deg", type=float, default=8.0)
    parser.add_argument("--gain_xy", type=float, default=0.9)
    parser.add_argument("--gain_yaw", type=float, default=0.9)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--pause", type=float, default=0.7)
    parser.add_argument("--tolerance_xy_mm", type=float, default=0.15)
    parser.add_argument("--tolerance_yaw_deg", type=float, default=0.3)
    parser.add_argument("--insert_depth_mm", type=float, default=14.0)
    parser.add_argument("--hold_seconds", type=float, default=20.0)
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
        obs = env.reset()
        p.resetDebugVisualizerCamera(
            cameraDistance=0.35,
            cameraYaw=-45,
            cameraPitch=-45,
            cameraTargetPosition=[-1.0, 0.0, 0.0],
        )
        p.addUserDebugText("start: misaligned peg", [-1.16, 0.0, 0.08], textSize=1.2, lifeTime=2)
        obs, _, _, _ = env.step([args.dx_mm / 1000.0, args.dy_mm / 1000.0, args.dyaw_deg])
        sleep_steps(args.pause * 2)

        for step in range(args.max_steps):
            dx, dy = obs["dxy"]
            dyaw = obs["dyaw"]
            xy_error_mm = math.hypot(dx, dy) * 1000.0

            p.addUserDebugText(
                f"align step {step}: {xy_error_mm:.2f}mm, {dyaw:.2f}deg",
                [-1.16, 0.0, 0.08],
                textSize=1.2,
                lifeTime=args.pause,
            )

            if xy_error_mm <= args.tolerance_xy_mm and abs(dyaw) <= args.tolerance_yaw_deg:
                break

            action = [-args.gain_xy * dx, -args.gain_xy * dy, -args.gain_yaw * dyaw]
            obs, _, _, _ = env.step(action)
            sleep_steps(args.pause)

        # Remove the last numerical residue so the visual demo starts the z push
        # from the exact nominal aligned pose.
        env.peg_dx_acc = 0.0
        env.peg_dy_acc = 0.0
        env.peg_dyaw_acc = 0.0
        env.position_control(180, env.init_position, env.init_orientation)
        obs_img, obs_gt = env.render()
        del obs_img, obs_gt
        sleep_steps(args.pause)

        p.addUserDebugText("centered: pushing down", [-1.16, 0.0, 0.08], textSize=1.2, lifeTime=2)

        start_z = env.init_position[2]
        target_z = start_z - args.insert_depth_mm / 1000.0
        steps = 12
        for i in range(steps):
            z = start_z + (target_z - start_z) * ((i + 1) / steps)
            env.position_control(
                90,
                np.array([env.init_position[0], env.init_position[1], z]),
                env.init_orientation,
            )
            sleep_steps(args.pause / 2)

        end_pos = p.getLinkState(env.panda_peg_id, env.endEffectorIndex)[0]
        print(
            "final end-effector pose approx: "
            f"x_error={(end_pos[0] + 1) * 1000:.3f}mm, "
            f"y_error={end_pos[1] * 1000:.3f}mm, "
            f"z={end_pos[2] * 1000:.3f}mm"
        )
        p.addUserDebugText("done", [-1.16, 0.0, 0.08], textSize=1.2, lifeTime=2)
        sleep_steps(args.hold_seconds)
    finally:
        env.close()


if __name__ == "__main__":
    main()
