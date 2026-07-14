"""Configuration objects for the Panda execution layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskToWorldTransform:
    """Planar task-frame to PyBullet world-frame transform.

    The first validated implementation intentionally keeps task +X/+Y aligned
    with world +X/+Y.  Keeping this as an explicit object makes command-sign
    tests the authority instead of burying assumptions in controller code.
    """

    origin_world: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    x_axis_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    y_axis_world: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def task_delta_to_world_delta(self, dx_m: float, dy_m: float) -> np.ndarray:
        return np.asarray(self.x_axis_world, dtype=np.float64) * float(dx_m) + np.asarray(
            self.y_axis_world, dtype=np.float64
        ) * float(dy_m)

    def task_pose_to_world_pos(self, dx_m: float, dy_m: float, z_m: float) -> np.ndarray:
        return (
            np.asarray(self.origin_world, dtype=np.float64)
            + self.task_delta_to_world_delta(dx_m, dy_m)
            + np.asarray([0.0, 0.0, float(z_m)], dtype=np.float64)
        )

    def world_delta_to_task_delta(self, delta_world: np.ndarray) -> np.ndarray:
        d = np.asarray(delta_world, dtype=np.float64).reshape(3)
        return np.asarray(
            [float(np.dot(d, self.x_axis_world)), float(np.dot(d, self.y_axis_world))],
            dtype=np.float64,
        )


@dataclass
class PandaConfig:
    gui: bool = False
    time_step: float = 1.0 / 240.0
    gravity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_urdf: str = "franka_panda/panda.urdf"
    robot_base_pos: tuple[float, float, float] = (-0.5, 0.0, 0.0)
    robot_base_orn_euler_deg: tuple[float, float, float] = (0.0, 0.0, 180.0)
    ee_link_index: int = 11
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    finger_joint_indices: tuple[int, ...] = (9, 10)
    rest_poses: tuple[float, ...] = (0.327, 0.369, -0.293, -2.383, 0.261, 2.726, 2.17)
    # The peg remains rigidly attached for deterministic controller testing,
    # but the jaws are closed around it so the simulated tool geometry matches
    # the visible grasp.  This is not a friction-based grasp model.
    finger_open: float = 0.01
    # Legacy combined Panda+peg nominal peg reference is z ~= 0.002 m.
    alignment_z_m: float = 0.002
    insertion_target_z_m: float = -0.010
    tool_roll_pitch_yaw_deg: tuple[float, float, float] = (180.0, 0.0, 90.0)
    # Seat the peg reference at panda_grasptarget.  The peg mesh extends along
    # local -Z, placing its upper section between the closed jaws instead of
    # leaving the legacy 66 mm visual gap below the gripper.
    attach_parent_frame_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attach_parent_frame_yaw_deg: float = -90.0
    # Compensate the Panda tool's 180-degree roll so the peg OBJ's local -Z
    # axis points downward into the fixture rather than upward through the EE.
    attach_child_frame_rpy_deg: tuple[float, float, float] = (0.0, 180.0, 0.0)
    peg_tip_offset_m: float = 0.035
    ik_max_iterations: int = 200
    ik_residual_threshold: float = 1e-8
    command_steps: int = 120
    position_gain: float = 0.08
    velocity_gain: float = 1.0
    max_force: float = 600.0
    peg_mass_kg: float = 0.1
    peg_lateral_friction: float = 0.5
    fixture_lateral_friction: float = 0.5
    peg_rgba: tuple[float, float, float, float] = (0.15, 0.35, 0.85, 1.0)
    collision_margin_m: float = 5e-5
    use_convex_decomposition: bool = True
    # ``kinematic`` preserves the reset-based coordinate/IK validation path.
    # ``dynamic`` uses motor simulation without post-command joint or peg
    # teleportation and is required for tracking/insertion claims.
    execution_mode: str = "kinematic"
    mesh_derived_alignment_z: bool = False
    alignment_clearance_mm: float = 0.5
    # When enabled, Panda env observations come from PyBullet camera RGB and
    # body-ID segmentation instead of the clean synthetic direct renderer.
    native_camera: bool = False
    # Optional eye-to-hand observation mode that excludes the robot from the
    # camera render.  It preserves the calibrated top-down perception view
    # when the real gripper geometry would occlude the seated peg.  Physics and
    # the interactive debug view are unaffected.
    camera_ignore_robot_occlusion: bool = False
    camera_eye_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.10)
    camera_target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up_vector: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.execution_mode not in {"kinematic", "dynamic"}:
            raise ValueError("execution_mode must be kinematic or dynamic")
        if self.alignment_clearance_mm < 0:
            raise ValueError("alignment_clearance_mm must be non-negative")
        if self.collision_margin_m < 0:
            raise ValueError("collision_margin_m must be non-negative")
        if self.peg_mass_kg <= 0 or self.peg_lateral_friction < 0 or self.fixture_lateral_friction < 0:
            raise ValueError("mass must be positive and friction must be non-negative")
        if len(self.peg_rgba) != 4 or any(v < 0 or v > 1 for v in self.peg_rgba):
            raise ValueError("peg_rgba must contain four values in [0,1]")


def default_asset_root() -> Path:
    return (Path(__file__).resolve().parents[2] / "gymEnv" / "envs" / "complex").resolve()
