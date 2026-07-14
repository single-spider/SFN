"""Gym-0.26 Panda alignment environment using measured Panda pose."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..config import CameraConfig, EnvironmentConfig
from ..envs.renderer import SyntheticDirectRenderer
from ..envs.spaces import gym, spaces
from ..geometry import dense_error_value, is_success, normalized_to_physical_action, xy_error_mm, yaw_error_deg
from .config import PandaConfig
from .panda_scene import PandaScene


class PandaPegInHoleAlignmentEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        shapes: list[str] | None = None,
        env_config: EnvironmentConfig | None = None,
        camera_config: CameraConfig | None = None,
        panda_config: PandaConfig | None = None,
        seed: int | None = None,
        motion_observer: Callable[[object, int, int], None] | None = None,
        motion_observer_stride: int = 8,
    ):
        super().__init__()
        self.config = env_config or EnvironmentConfig()
        self.camera_config = camera_config or CameraConfig()
        self.panda_config = panda_config or PandaConfig()
        self.shapes = list(shapes or ["square-concave1"])
        self._seed = int(seed or 0)
        self.rng = np.random.default_rng(self._seed)
        self.renderer = SyntheticDirectRenderer(self.camera_config)
        self.scene: PandaScene | None = None
        self.step_count = 0
        self._last_E = 0.0
        if motion_observer is not None and not callable(motion_observer):
            raise TypeError("motion_observer must be callable")
        if motion_observer_stride < 1:
            raise ValueError("motion_observer_stride must be at least one")
        self.motion_observer = motion_observer
        self.motion_observer_stride = int(motion_observer_stride)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    0, 255, shape=(3, self.camera_config.crop_height, self.camera_config.crop_width), dtype=np.uint8
                ),
                "mask": spaces.Box(
                    0, 2, shape=(self.camera_config.crop_height, self.camera_config.crop_width), dtype=np.uint8
                ),
                "pose_error": spaces.Box(
                    low=np.asarray(
                        [
                            -self.config.xy_workspace_mm / 1000.0,
                            -self.config.xy_workspace_mm / 1000.0,
                            -self.config.yaw_workspace_deg,
                        ],
                        dtype=np.float32,
                    ),
                    high=np.asarray(
                        [
                            self.config.xy_workspace_mm / 1000.0,
                            self.config.xy_workspace_mm / 1000.0,
                            self.config.yaw_workspace_deg,
                        ],
                        dtype=np.float32,
                    ),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "joint_positions": spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                "ee_pose": spaces.Box(
                    low=np.asarray([-10, -10, -10, -1, -1, -1, -1], dtype=np.float32),
                    high=np.asarray([10, 10, 10, 1, 1, 1, 1], dtype=np.float32),
                ),
                "peg_pose": spaces.Box(
                    low=np.asarray([-10, -10, -10, -1, -1, -1, -1], dtype=np.float32),
                    high=np.asarray([10, 10, 10, 1, 1, 1, 1], dtype=np.float32),
                ),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        options = options or {}
        shape = str(options.get("shape") or self.rng.choice(self.shapes))
        pose_error = np.asarray(options.get("pose_error", self._sample_pose_error(options)), dtype=np.float32).reshape(
            3
        )
        if self.scene is not None:
            self.scene.close()
        self.scene = PandaScene(shape=shape, config=self.panda_config, seed=self._seed)
        self.scene.reset_to_pose_error(pose_error)
        self.step_count = 0
        self._last_E = self._error_value(pose_error)
        return self._make_obs(), self._make_info(None, None, None, False)

    def _sample_pose_error(self, options: dict) -> np.ndarray:
        er = options.get("error_range") or {}
        xy_low, xy_high = -self.config.xy_initial_range_mm / 1000.0, self.config.xy_initial_range_mm / 1000.0
        yaw_low, yaw_high = -self.config.yaw_initial_range_deg, self.config.yaw_initial_range_deg
        if "xy_m" in er:
            lo, hi = map(float, er["xy_m"])
            xy_low, xy_high = (-hi, hi) if lo >= 0 else (lo, hi)
        if "yaw_deg" in er:
            lo, hi = map(float, er["yaw_deg"])
            yaw_low, yaw_high = (-hi, hi) if lo >= 0 else (lo, hi)
        for _ in range(1000):
            pose = np.asarray(
                [
                    self.rng.uniform(xy_low, xy_high),
                    self.rng.uniform(xy_low, xy_high),
                    self.rng.uniform(yaw_low, yaw_high),
                ],
                dtype=np.float32,
            )
            if not options.get("nontrivial", True) or not is_success(
                pose, self.config.xy_success_axis_mm, self.config.yaw_success_deg
            ):
                return pose
        return pose

    def step(self, action):
        if self.scene is None:
            raise RuntimeError("reset() must be called before step()")
        before_pose = self.scene.measure().pose_error_task.copy()
        action_norm = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
        action_phys = normalized_to_physical_action(
            action_norm, self.config.max_action_xy_mm, self.config.max_action_yaw_deg
        )
        old_E = self._last_E
        result = self.scene.execute_cartesian_delta(
            float(action_phys[0]),
            float(action_phys[1]),
            float(action_phys[2]),
            physics_observer=self.motion_observer,
            observer_stride=self.motion_observer_stride,
        )
        self.step_count += 1
        pose = self.scene.measure().pose_error_task.astype(np.float32)
        current_E = self._error_value(pose)
        self._last_E = current_E
        success = is_success(pose, self.config.xy_success_axis_mm, self.config.yaw_success_deg)
        terminated = bool(success)
        truncated = bool((not terminated) and self.step_count >= self.config.max_steps)
        reward = float(old_E - current_E - self.config.step_penalty)
        if success:
            reward += self.config.success_bonus
        info = self._make_info(
            action_norm, action_phys, result, False, before_pose=before_pose
        )
        if truncated:
            info["failure_state"] = "panda_alignment_timeout"
        return self._make_obs(), reward, terminated, truncated, info

    def _error_value(self, pose):
        return dense_error_value(
            pose,
            self.config.xy_workspace_mm,
            self.config.yaw_workspace_deg,
            self.config.reward_w_xy,
            self.config.reward_w_yaw,
        )

    def _make_obs(self):
        assert self.scene is not None
        m = self.scene.measure()
        rendered = (
            self.scene.render_camera(self.camera_config)
            if self.panda_config.native_camera
            else self.renderer.render(m.pose_error_task.astype(np.float32), self.scene.shape, self.rng)
        )
        return {
            "rgb": rendered.rgb,
            "mask": rendered.mask,
            "pose_error": m.pose_error_task.astype(np.float32),
            "joint_positions": m.joint_positions.astype(np.float32),
            "ee_pose": np.asarray([*m.ee_pos_world, *m.ee_quat_world], dtype=np.float32),
            "peg_pose": np.asarray([*m.peg_pos_world, *m.peg_quat_world], dtype=np.float32),
        }

    def _make_info(self, action_normalized, action_physical, result, out_of_bounds: bool, before_pose=None):
        assert self.scene is not None
        m = self.scene.measure()
        pose = m.pose_error_task
        contacts = list(self.scene.contact_summary() if result is None else result.contacts)
        measured_action = None
        if before_pose is not None:
            measured_action = pose - np.asarray(before_pose, dtype=np.float64)
            measured_action[2] = ((float(measured_action[2]) + 180.0) % 360.0) - 180.0
        margins = self.scene.joint_limit_margins(m.joint_positions)
        max_force = max((float(c.get("normal_force", 0.0)) for c in contacts), default=0.0)
        max_penetration_mm = max((max(0.0, -float(c.get("distance", 0.0))) * 1000.0 for c in contacts), default=0.0)
        success = is_success(pose, self.config.xy_success_axis_mm, self.config.yaw_success_deg)
        return {
            "shape": self.scene.shape,
            "seed": self._seed,
            "step": self.step_count,
            "pose_error": pose.astype(np.float32),
            "xy_error_mm": xy_error_mm(pose),
            "yaw_error_deg": yaw_error_deg(pose),
            "success": bool(success),
            "ik_success": True if result is None else bool(result.ik_success),
            "tracking_error_mm": 0.0 if result is None else float(result.pos_error_mm),
            "tracking_yaw_error_deg": 0.0 if result is None else float(result.yaw_error_deg),
            "joint_positions": m.joint_positions.astype(np.float32),
            "joint_target": None if result is None else result.joint_target.astype(np.float32),
            "joint_limit_margins": margins.astype(np.float32),
            "min_joint_limit_margin": float(np.min(margins)),
            "joint_limit_violation": bool(np.any(margins < -1e-7)),
            "ik_residual_m": None if result is None else result.ik_residual_m,
            "ik_branch": None if result is None else result.ik_branch,
            "contact_count": len(contacts),
            "contacts": contacts,
            "max_contact_force": max_force,
            "max_penetration_mm": max_penetration_mm,
            "unexpected_contact": False,
            "out_of_bounds": bool(out_of_bounds),
            "action_normalized": None if action_normalized is None else np.asarray(action_normalized, dtype=np.float32),
            "action_physical": None if action_physical is None else np.asarray(action_physical, dtype=np.float32),
            "measured_action_physical": None if measured_action is None else measured_action.astype(np.float32),
            "commanded_peg_pose": None if result is None else result.commanded_peg_pose.astype(np.float32),
            "measured_peg_pose": np.asarray([*m.peg_pos_world, *m.peg_quat_world], dtype=np.float32),
            "lateral_drift_mm": 0.0,
            "insertion_depth_mm": None,
            "failure_state": None,
        }

    def render(self):
        return self._make_obs()["rgb"]

    def close(self):
        if self.scene is not None:
            self.scene.close()
            self.scene = None
        self.renderer.close()
