"""Deterministic Gym-0.26 style peg-in-hole alignment environment."""

from __future__ import annotations

import numpy as np

from ..config import CameraConfig, EnvironmentConfig
from ..geometry import dense_error_value, is_success, normalized_to_physical_action, xy_error_mm, yaw_error_deg
from .asset_registry import AssetRegistry
from .renderer import make_renderer
from .scene import SceneState
from .spaces import gym, spaces


class PegInHoleAlignmentEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_config: EnvironmentConfig | None = None,
        camera_config: CameraConfig | None = None,
        asset_registry: AssetRegistry | None = None,
        shapes: list[str] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.config = env_config or EnvironmentConfig()
        self.camera_config = camera_config or CameraConfig()
        self.registry = asset_registry or AssetRegistry()
        self.shapes = list(shapes) if shapes is not None else self.registry.list_shapes()
        if not self.shapes:
            self.shapes = ["synthetic-square"]
        self._seed = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)
        self.renderer = make_renderer(self.camera_config, self.registry)
        self.state: SceneState | None = None
        self._last_E = 0.0
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
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        options = options or {}
        shape = options.get("shape")
        if shape is None:
            shape = str(self.rng.choice(self.shapes))
        if shape not in self.shapes and shape != "synthetic-square":
            raise KeyError(f"Requested shape {shape!r} not available")
        pose_error = options.get("pose_error")
        if pose_error is None:
            pose_error = self._sample_pose_error(options)
        pose = self._clip_pose(np.asarray(pose_error, dtype=np.float32).reshape(3))
        self.state = SceneState(str(shape), pose, 0)
        self._last_E = self._error_value(pose)
        obs = self._make_obs()
        info = self._make_info(None, None, False)
        return obs, info

    def _sample_pose_error(self, options: dict) -> np.ndarray:
        er = options.get("error_range") or {}
        xy = er.get("xy_m")
        yaw = er.get("yaw_deg")
        if xy is None:
            xy_low, xy_high = -self.config.xy_initial_range_mm / 1000.0, self.config.xy_initial_range_mm / 1000.0
        else:
            lo, hi = map(float, xy)
            xy_low, xy_high = (-hi, hi) if lo >= 0 else (lo, hi)
        if yaw is None:
            yaw_low, yaw_high = -self.config.yaw_initial_range_deg, self.config.yaw_initial_range_deg
        else:
            lo, hi = map(float, yaw)
            yaw_low, yaw_high = (-hi, hi) if lo >= 0 else (lo, hi)
        pose = np.zeros(3, dtype=np.float32)
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
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        action_norm = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
        action_phys = normalized_to_physical_action(
            action_norm, self.config.max_action_xy_mm, self.config.max_action_yaw_deg
        )
        old_E = self._last_E
        unclipped = self.state.pose_error + action_phys
        new_pose = self._clip_pose(unclipped)
        out_of_bounds = not np.allclose(unclipped, new_pose)
        self.state.pose_error = new_pose.astype(np.float32)
        self.state.step_count += 1
        current_E = self._error_value(self.state.pose_error)
        self._last_E = current_E
        success = is_success(self.state.pose_error, self.config.xy_success_axis_mm, self.config.yaw_success_deg)
        terminated = bool(success or out_of_bounds)
        truncated = bool((not terminated) and self.state.step_count >= self.config.max_steps)
        reward = float(old_E - current_E - self.config.step_penalty)
        if success:
            reward += self.config.success_bonus
        if out_of_bounds:
            reward -= self.config.failure_penalty
        return self._make_obs(), reward, terminated, truncated, self._make_info(action_norm, action_phys, out_of_bounds)

    def _error_value(self, pose):
        return dense_error_value(
            pose,
            self.config.xy_workspace_mm,
            self.config.yaw_workspace_deg,
            self.config.reward_w_xy,
            self.config.reward_w_yaw,
        )

    def _clip_pose(self, pose):
        xy = self.config.xy_workspace_mm / 1000.0
        return np.asarray(
            [
                np.clip(pose[0], -xy, xy),
                np.clip(pose[1], -xy, xy),
                np.clip(pose[2], -self.config.yaw_workspace_deg, self.config.yaw_workspace_deg),
            ],
            dtype=np.float32,
        )

    def _make_obs(self):
        assert self.state is not None
        rendered = self.renderer.render(self.state.pose_error, self.state.shape, self.rng)
        return {
            "rgb": rendered.rgb,
            "mask": rendered.mask,
            "pose_error": self.state.pose_error.astype(np.float32, copy=True),
        }

    def _make_info(self, action_normalized, action_physical, out_of_bounds: bool):
        assert self.state is not None
        success = is_success(self.state.pose_error, self.config.xy_success_axis_mm, self.config.yaw_success_deg)
        return {
            "shape": self.state.shape,
            "seed": self._seed,
            "step": self.state.step_count,
            "pose_error": self.state.pose_error.astype(np.float32, copy=True),
            "xy_error_mm": xy_error_mm(self.state.pose_error),
            "yaw_error_deg": yaw_error_deg(self.state.pose_error),
            "success": bool(success),
            "out_of_bounds": bool(out_of_bounds),
            "action_normalized": None if action_normalized is None else np.asarray(action_normalized, dtype=np.float32),
            "action_physical": None if action_physical is None else np.asarray(action_physical, dtype=np.float32),
        }

    def render(self):
        return self._make_obs()["rgb"]

    def close(self):
        self.renderer.close()
