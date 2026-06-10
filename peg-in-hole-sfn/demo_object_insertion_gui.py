import argparse
import math
import os
import time

import pybullet as p
import pybullet_data


def step_sleep(seconds):
    end = time.time() + seconds
    while time.time() < end:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peg_type", default="square-concave1")
    parser.add_argument("--dx_mm", type=float, default=8.0)
    parser.add_argument("--dy_mm", type=float, default=-6.0)
    parser.add_argument("--dyaw_deg", type=float, default=12.0)
    parser.add_argument("--start_z_mm", type=float, default=35.0)
    parser.add_argument("--insert_z_mm", type=float, default=-12.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--hold_seconds", type=float, default=60.0)
    args = parser.parse_args()

    root = os.path.dirname(os.path.realpath(__file__))
    shape_dir = os.path.join(root, "gymEnv", "envs", "complex", args.peg_type)
    base_urdf = os.path.join(shape_dir, "base", "base.urdf")
    peg_urdf = os.path.join(shape_dir, "peg", "peg_test.urdf")

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(shape_dir)
    p.setGravity(0, 0, -9.8)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.18,
        cameraYaw=-35,
        cameraPitch=-55,
        cameraTargetPosition=[0, 0, 0],
    )

    base_id = p.loadURDF(base_urdf, basePosition=[0, 0, 0], useFixedBase=True)
    peg_id = p.loadURDF(
        peg_urdf,
        basePosition=[args.dx_mm / 1000.0, args.dy_mm / 1000.0, args.start_z_mm / 1000.0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, math.radians(args.dyaw_deg)]),
        useFixedBase=True,
    )
    p.changeVisualShape(base_id, -1, rgbaColor=[0.85, 0.85, 0.85, 1.0])
    p.changeVisualShape(peg_id, -1, rgbaColor=[0.0, 0.8, 0.1, 1.0])

    p.addUserDebugText("misaligned", [-0.035, 0, 0.04], textSize=1.1, lifeTime=2)
    step_sleep(1.5)

    for i in range(args.steps):
        t = (i + 1) / args.steps
        smooth_t = t * t * (3 - 2 * t)
        x = lerp(args.dx_mm / 1000.0, 0.0, smooth_t)
        y = lerp(args.dy_mm / 1000.0, 0.0, smooth_t)
        yaw = lerp(math.radians(args.dyaw_deg), 0.0, smooth_t)
        p.resetBasePositionAndOrientation(
            peg_id,
            [x, y, args.start_z_mm / 1000.0],
            p.getQuaternionFromEuler([0, 0, yaw]),
        )
        p.stepSimulation()
        time.sleep(1.0 / 120.0)

    p.addUserDebugText("aligned", [-0.035, 0, 0.04], textSize=1.1, lifeTime=2)
    step_sleep(0.8)

    for i in range(args.steps):
        t = (i + 1) / args.steps
        smooth_t = t * t * (3 - 2 * t)
        z = lerp(args.start_z_mm / 1000.0, args.insert_z_mm / 1000.0, smooth_t)
        p.resetBasePositionAndOrientation(peg_id, [0, 0, z], p.getQuaternionFromEuler([0, 0, 0]))
        p.stepSimulation()
        time.sleep(1.0 / 120.0)

    p.addUserDebugText("inserted visual", [-0.035, 0, 0.04], textSize=1.1, lifeTime=2)
    step_sleep(args.hold_seconds)
    p.disconnect()


if __name__ == "__main__":
    main()
