"""Deterministic camera and command disturbances for paired evaluation.

Camera-only profiles alter executable image-formation effects before RGB
segmentation and never edit semantic labels. Legacy diagnostic profiles remain
available under explicit names for historical ablations. Action profiles model
noise, backlash, delay, calibration bias, and attachment bias independently.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from ..constants import MASK_BACKGROUND, MASK_SEAM
from ..geometry import decode_position
from ..models.vsn import VirtualSensorNetwork, VSNOutput


@dataclass(frozen=True)
class DisturbanceConfig:
    """Disturbances applied before VSN position/orientation inference.

    Values are intentionally simple and image-size independent where possible:

    - ``rgb_noise_std`` is a fraction of the 0..255 RGB range.
    - ``mask_shift_px`` shifts the mask by up to this many pixels in X and Y.
    - ``seam_dropout_prob`` randomly removes seam pixels.
    - ``occlusion_prob`` controls whether a rectangular occluder is applied.
    - ``occlusion_frac`` is the approximate occluder size as a fraction of image
      width/height.
    - ``label_flip_prob`` randomly corrupts mask labels.
    """

    name: str = "clean"
    rgb_noise_std: float = 0.0
    mask_shift_px: int = 0
    seam_dropout_prob: float = 0.0
    occlusion_prob: float = 0.0
    occlusion_frac: float = 0.0
    label_flip_prob: float = 0.0
    occlusion_burst_length: int = 0
    occlusion_burst_period: int = 8
    blur_kernel_px: int = 0
    motion_blur_px: int = 0
    gamma: float = 1.0
    white_balance_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intrinsic_scale: float = 1.0
    crop_fraction: float = 0.0
    lighting_gain: float = 1.0
    lighting_bias: float = 0.0
    lighting_gradient: float = 0.0
    camera_only: bool = False
    seed: int = 1

    def __post_init__(self) -> None:
        if self.blur_kernel_px < 0 or self.motion_blur_px < 0:
            raise ValueError("blur kernel lengths must be non-negative")
        if self.gamma <= 0 or self.intrinsic_scale <= 0:
            raise ValueError("gamma and intrinsic_scale must be positive")
        if not 0.0 <= self.crop_fraction < 1.0:
            raise ValueError("crop_fraction must be in [0, 1)")
        if len(self.white_balance_rgb) != 3 or any(float(value) < 0 for value in self.white_balance_rgb):
            raise ValueError("white_balance_rgb must contain three non-negative gains")

    def to_dict(self) -> dict:
        return asdict(self)


ROBUSTNESS_PROFILES: dict[str, DisturbanceConfig] = {
    "clean": DisturbanceConfig(name="clean"),
    "rgb_noise": DisturbanceConfig(name="rgb_noise", rgb_noise_std=0.04),
    "mask_shift": DisturbanceConfig(name="mask_shift", mask_shift_px=3),
    "seam_dropout": DisturbanceConfig(name="seam_dropout", seam_dropout_prob=0.35),
    "occlusion": DisturbanceConfig(name="occlusion", occlusion_prob=1.0, occlusion_frac=0.18),
    "occlusion_burst3": DisturbanceConfig(name="occlusion_burst3", occlusion_frac=0.22, occlusion_burst_length=3),
    "occlusion_burst5": DisturbanceConfig(name="occlusion_burst5", occlusion_frac=0.22, occlusion_burst_length=5),
    "combined": DisturbanceConfig(
        name="combined",
        rgb_noise_std=0.03,
        mask_shift_px=2,
        seam_dropout_prob=0.25,
        occlusion_prob=0.6,
        occlusion_frac=0.14,
        label_flip_prob=0.002,
    ),
    "blur": DisturbanceConfig(name="blur", blur_kernel_px=5, camera_only=True),
    "motion_blur": DisturbanceConfig(name="motion_blur", motion_blur_px=7, camera_only=True),
    "gamma": DisturbanceConfig(name="gamma", gamma=1.45, camera_only=True),
    "white_balance": DisturbanceConfig(
        name="white_balance", white_balance_rgb=(1.16, 1.0, 0.84), camera_only=True
    ),
    "intrinsic_scale": DisturbanceConfig(name="intrinsic_scale", intrinsic_scale=1.12, camera_only=True),
    "crop": DisturbanceConfig(name="crop", crop_fraction=0.12, camera_only=True),
    "lighting": DisturbanceConfig(
        name="lighting", lighting_gain=0.72, lighting_bias=0.06, lighting_gradient=0.18, camera_only=True
    ),
}

# Locked monotonic combined-disturbance curve.  Level zero is clean; later
# levels scale camera-executable image corruption rather than changing labels,
# the controller, or the success criterion.  ``camera_only`` ensures predicted
# mask runs corrupt RGB before segmentation instead of editing labels afterward.
for _level, _scale in enumerate((0.0, 0.5, 1.0, 1.75, 2.5)):
    ROBUSTNESS_PROFILES[f"severity_{_level}"] = DisturbanceConfig(
        name=f"severity_{_level}",
        rgb_noise_std=0.04 * _scale,
        mask_shift_px=int(round(3 * _scale)),
        occlusion_prob=min(1.0, 0.40 * _scale),
        occlusion_frac=0.0 if _level == 0 else 0.16 * min(2.5, _scale),
        camera_only=True,
    )


@dataclass(frozen=True)
class ActionDisturbanceConfig:
    """Normalized action/command transmission disturbances.

    The three axes are ``(x, y, yaw)`` in the environment's normalized action
    units. ``calibration_offset`` and ``attachment_offset`` are kept separate
    in metadata even though both become additive command-frame biases.  Delay
    is measured in control steps and initial delayed commands are zero.
    """

    name: str = "clean_action"
    noise_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    backlash: tuple[float, float, float] = (0.0, 0.0, 0.0)
    delay_steps: int = 0
    calibration_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attachment_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clip_commands: bool = True
    seed: int = 1

    def __post_init__(self) -> None:
        for field_name in ("noise_std", "backlash", "calibration_offset", "attachment_offset"):
            values = getattr(self, field_name)
            if len(values) != 3:
                raise ValueError(f"{field_name} must contain x, y, and yaw values")
        if any(float(value) < 0 for value in self.noise_std):
            raise ValueError("noise_std must be non-negative")
        if any(float(value) < 0 for value in self.backlash):
            raise ValueError("backlash must be non-negative")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)


ACTION_DISTURBANCE_PROFILES: dict[str, ActionDisturbanceConfig] = {
    "clean_action": ActionDisturbanceConfig(),
    "command_noise": ActionDisturbanceConfig(name="command_noise", noise_std=(0.04, 0.04, 0.025)),
    "command_backlash": ActionDisturbanceConfig(name="command_backlash", backlash=(0.06, 0.06, 0.03)),
    "command_delay": ActionDisturbanceConfig(name="command_delay", delay_steps=2),
    "calibration_offset": ActionDisturbanceConfig(
        name="calibration_offset", calibration_offset=(0.04, -0.03, 0.025)
    ),
    "attachment_offset": ActionDisturbanceConfig(
        name="attachment_offset", attachment_offset=(-0.025, 0.035, -0.02)
    ),
    "combined_command": ActionDisturbanceConfig(
        name="combined_command",
        noise_std=(0.025, 0.025, 0.015),
        backlash=(0.04, 0.04, 0.02),
        delay_steps=1,
        calibration_offset=(0.02, -0.015, 0.01),
        attachment_offset=(-0.01, 0.015, -0.01),
    ),
}


def disturb_action(
    action: Sequence[float] | np.ndarray,
    config: ActionDisturbanceConfig,
    *,
    episode_seed: int,
    frame_index: int,
) -> np.ndarray:
    """Apply stateless, episode/frame-keyed command noise, bias, and deadband.

    Use :class:`ActionDisturbance` when delay state is required.  Backlash is a
    per-axis lost-motion deadband: command magnitude inside the configured
    width is lost, and larger commands are reduced by that width.
    """

    command = np.asarray(action, dtype=np.float32).reshape(3).astype(np.float64)
    rng = np.random.default_rng(np.random.SeedSequence([int(episode_seed), int(frame_index), 0xAC710]))
    command += rng.normal(0.0, np.asarray(config.noise_std, dtype=np.float64), size=3)
    command += np.asarray(config.calibration_offset, dtype=np.float64)
    command += np.asarray(config.attachment_offset, dtype=np.float64)
    deadband = np.asarray(config.backlash, dtype=np.float64)
    command = np.sign(command) * np.maximum(np.abs(command) - deadband, 0.0)
    if config.clip_commands:
        command = np.clip(command, -1.0, 1.0)
    return command.astype(np.float32)


class ActionDisturbance:
    """Stateful deterministic command channel supporting fixed-step delay."""

    def __init__(self, config: ActionDisturbanceConfig):
        self.config = config
        self._episode_seed = int(config.seed)
        self._frame_index = 0
        self._queue: deque[np.ndarray] = deque()

    def set_episode_seed(self, seed: int) -> None:
        self._episode_seed = int(seed)
        self.reset_state()

    def reset_state(self) -> None:
        self._frame_index = 0
        self._queue.clear()

    def apply(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        disturbed = disturb_action(
            action, self.config, episode_seed=self._episode_seed, frame_index=self._frame_index
        )
        self._frame_index += 1
        self._queue.append(disturbed)
        if len(self._queue) <= int(self.config.delay_steps):
            return np.zeros(3, dtype=np.float32)
        return self._queue.popleft()

    __call__ = apply


class DisturbedActionEnv:
    """Transparent environment wrapper that disturbs actions before ``step``.

    This intentionally relies only on the standard ``reset``/``step`` contract
    so it can wrap the synthetic and Panda environments without either module
    importing evaluation code.
    """

    def __init__(self, env, config: ActionDisturbanceConfig):
        self.env = env
        self.action_disturbance = ActionDisturbance(config)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        self.action_disturbance.set_episode_seed(self.action_disturbance.config.seed if seed is None else int(seed))
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float32).reshape(3)
        executed = self.action_disturbance.apply(commanded)
        observation, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info["action_disturbance"] = self.action_disturbance.config.to_dict()
        info["action_commanded_normalized"] = commanded.copy()
        info["action_executed_normalized"] = executed.copy()
        return observation, reward, terminated, truncated, info


def parse_profile_names(text: str | None) -> list[str]:
    if not text:
        return ["clean", "rgb_noise", "mask_shift", "seam_dropout", "occlusion", "combined"]
    names = [x.strip() for x in text.split(",") if x.strip()]
    unknown = [x for x in names if x not in ROBUSTNESS_PROFILES]
    if unknown:
        raise ValueError(f"Unknown robustness profile(s): {', '.join(unknown)}")
    return names


def disturb_observation(obs: dict, config: DisturbanceConfig, *, episode_seed: int, frame_index: int) -> dict:
    """Apply episode/frame-keyed corruption directly to one observation.

    Camera-only configurations never mutate ``obs["mask"]``.  This makes the
    camera families executable against a real camera while retaining the mask
    solely as evaluation metadata.  Legacy non-camera-only profiles continue
    to perturb RGB and labels together.
    """
    result = dict(obs)
    rgb = np.asarray(obs["rgb"]).copy()
    mask = np.asarray(obs["mask"]).copy()
    rng = np.random.default_rng(np.random.SeedSequence([int(episode_seed), int(frame_index)]))
    scale = 255.0 if float(rgb.max()) > 3.0 else 1.0
    if config.rgb_noise_std > 0:
        rgb = np.clip(rgb.astype(np.float32) + rng.normal(0, config.rgb_noise_std * scale, rgb.shape), 0, scale)
        rgb = rgb.astype(obs["rgb"].dtype)
    if config.mask_shift_px > 0:
        dx = int(rng.integers(-config.mask_shift_px, config.mask_shift_px + 1))
        dy = int(rng.integers(-config.mask_shift_px, config.mask_shift_px + 1))
        if not config.camera_only:
            mask = _shift_zero_fill(mask, dx, dy)
        shifted_rgb = np.zeros_like(rgb)
        for channel in range(rgb.shape[0]):
            shifted_rgb[channel] = _shift_zero_fill(rgb[channel], dx, dy)
        rgb = shifted_rgb
    if config.seam_dropout_prob > 0 and not config.camera_only:
        seam = mask == MASK_SEAM
        mask[seam & (rng.random(mask.shape) < config.seam_dropout_prob)] = MASK_BACKGROUND
    burst = False
    if config.occlusion_burst_length > 0:
        period = max(config.occlusion_burst_length + 1, config.occlusion_burst_period)
        burst = frame_index % period < config.occlusion_burst_length
    if burst or (config.occlusion_prob > 0 and rng.random() < config.occlusion_prob):
        fraction = max(0.0, config.occlusion_frac)
        height, width = mask.shape
        oh, ow = max(1, round(height * fraction)), max(1, round(width * fraction))
        y0 = int(rng.integers(0, max(1, height - oh + 1)))
        x0 = int(rng.integers(0, max(1, width - ow + 1)))
        if not config.camera_only:
            mask[y0 : y0 + oh, x0 : x0 + ow] = MASK_BACKGROUND
        rgb[:, y0 : y0 + oh, x0 : x0 + ow] = 0
    if config.label_flip_prob > 0 and not config.camera_only:
        flip = rng.random(mask.shape) < config.label_flip_prob
        mask[flip] = rng.integers(0, 3, int(flip.sum()), dtype=np.uint8)
    rgb = _apply_camera_effects(rgb, config, rng, scale=scale, output_dtype=np.asarray(obs["rgb"]).dtype)
    result["rgb"] = rgb
    result["mask"] = mask
    result["disturbance"] = config.to_dict()
    return result


class DisturbedVirtualSensorNetwork(VirtualSensorNetwork):
    """Wrap a VSN and perturb RGB/masks before downstream inference."""

    def __init__(self, base: VirtualSensorNetwork, config: DisturbanceConfig):
        super().__init__(segmentation=base.segmentation, position=base.position, orientation=base.orientation)
        self.disturbance_config = config
        self._rng = np.random.default_rng(int(config.seed))
        self._episode_seed = int(config.seed)
        self._frame_index = 0

    def set_episode_seed(self, seed: int) -> None:
        """Lock disturbances to an episode key, independent of method order."""
        self._episode_seed = int(seed)
        self._rng = np.random.default_rng(self._episode_seed)
        self._frame_index = 0

    def reset_state(self) -> None:
        self._rng = np.random.default_rng(self._episode_seed)
        self._frame_index = 0

    def forward(self, rgb=None, mask=None):  # noqa: D401 - inherited contract
        # Derive corruption from (episode, frame), not from how many random
        # draws a prior controller happened to consume.
        self._rng = np.random.default_rng(np.random.SeedSequence([self._episode_seed, self._frame_index]))
        cfg = self.disturbance_config
        if (rgb is None) == (mask is None):
            raise ValueError("Exactly one of rgb or mask is required")
        if rgb is not None:
            if cfg.rgb_noise_std > 0:
                import torch

                scale = 255.0 if float(rgb.max().detach().cpu()) > 3.0 else 1.0
                gen = torch.Generator(device=rgb.device)
                gen.manual_seed(int(self._rng.integers(0, 2**31 - 1)))
                noise = (
                    torch.randn(rgb.shape, dtype=torch.float32, device=rgb.device, generator=gen)
                    * float(cfg.rgb_noise_std)
                    * scale
                )
                rgb = torch.clamp(rgb.float() + noise, 0.0, scale)
            if cfg.camera_only:
                rgb = self._disturb_camera_rgb(rgb)
            base_out = super().forward(rgb=rgb)
            if cfg.camera_only:
                self._frame_index += 1
                return base_out
            mask = base_out.mask
        if cfg.camera_only:
            # A mask-only caller has no camera signal on which a physically
            # executable disturbance can act. Preserve labels rather than
            # silently turning the camera profile into label corruption.
            self._frame_index += 1
            return super().forward(mask=mask)
        disturbed = self._disturb_mask(mask, frame_index=self._frame_index)
        self._frame_index += 1
        return super().forward(mask=disturbed)

    def _disturb_camera_rgb(self, rgb):
        """Apply camera-executable effects to RGB before segmentation."""
        import torch

        cfg = self.disturbance_config
        array = rgb.detach().cpu().numpy().copy()
        squeeze = array.ndim == 3
        if squeeze:
            array = array[None, ...]
        for index in range(array.shape[0]):
            image = array[index]
            if cfg.mask_shift_px > 0:
                dx = int(self._rng.integers(-cfg.mask_shift_px, cfg.mask_shift_px + 1))
                dy = int(self._rng.integers(-cfg.mask_shift_px, cfg.mask_shift_px + 1))
                shifted = np.zeros_like(image)
                for channel in range(image.shape[0]):
                    shifted[channel] = _shift_zero_fill(image[channel], dx, dy)
                image = shifted
            if cfg.occlusion_prob > 0 and self._rng.random() < float(cfg.occlusion_prob):
                height, width = image.shape[-2:]
                fraction = max(0.0, float(cfg.occlusion_frac))
                oh, ow = max(1, round(height * fraction)), max(1, round(width * fraction))
                y0 = int(self._rng.integers(0, max(1, height - oh + 1)))
                x0 = int(self._rng.integers(0, max(1, width - ow + 1)))
                image[:, y0 : y0 + oh, x0 : x0 + ow] = 0
            scale = 255.0 if float(np.max(image)) > 3.0 else 1.0
            array[index] = _apply_camera_effects(
                image, cfg, self._rng, scale=scale, output_dtype=array.dtype
            )
        if squeeze:
            array = array[0]
        return torch.as_tensor(array, dtype=rgb.dtype, device=rgb.device)

    def _disturb_mask(self, mask, frame_index: int = 0):
        cfg = self.disturbance_config
        if (
            cfg.mask_shift_px <= 0
            and cfg.seam_dropout_prob <= 0
            and cfg.occlusion_prob <= 0
            and cfg.label_flip_prob <= 0
            and cfg.occlusion_burst_length <= 0
        ):
            return mask

        import torch

        device = mask.device
        arr = mask.detach().cpu().numpy().astype(np.uint8, copy=True)
        for i in range(arr.shape[0]):
            m = arr[i]
            h, w = m.shape
            if cfg.mask_shift_px > 0:
                dx = int(self._rng.integers(-cfg.mask_shift_px, cfg.mask_shift_px + 1))
                dy = int(self._rng.integers(-cfg.mask_shift_px, cfg.mask_shift_px + 1))
                m = _shift_zero_fill(m, dx=dx, dy=dy)
            if cfg.seam_dropout_prob > 0:
                seam = m == MASK_SEAM
                drop = self._rng.random(m.shape) < float(cfg.seam_dropout_prob)
                m[seam & drop] = MASK_BACKGROUND
            if cfg.occlusion_prob > 0 and self._rng.random() < float(cfg.occlusion_prob):
                frac = max(0.0, float(cfg.occlusion_frac))
                oh = max(1, int(round(h * frac)))
                ow = max(1, int(round(w * frac)))
                y0 = int(self._rng.integers(0, max(1, h - oh + 1)))
                x0 = int(self._rng.integers(0, max(1, w - ow + 1)))
                m[y0 : y0 + oh, x0 : x0 + ow] = MASK_BACKGROUND
            if cfg.occlusion_burst_length > 0:
                period = max(int(cfg.occlusion_burst_length) + 1, int(cfg.occlusion_burst_period))
                phase = int(frame_index) % period
                if phase < int(cfg.occlusion_burst_length):
                    frac = max(0.0, float(cfg.occlusion_frac))
                    oh, ow = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
                    # Keep the rectangle fixed throughout one burst.
                    burst_rng = np.random.default_rng(
                        np.random.SeedSequence([self._episode_seed, int(frame_index) // period, i, 0xB017])
                    )
                    y0 = int(burst_rng.integers(0, max(1, h - oh + 1)))
                    x0 = int(burst_rng.integers(0, max(1, w - ow + 1)))
                    m[y0 : y0 + oh, x0 : x0 + ow] = MASK_BACKGROUND
            if cfg.label_flip_prob > 0:
                flip = self._rng.random(m.shape) < float(cfg.label_flip_prob)
                m[flip] = self._rng.integers(0, 3, size=int(flip.sum()), dtype=np.uint8)
            arr[i] = m
        return torch.as_tensor(arr, dtype=torch.long, device=device)


class TemporalSmoothedVirtualSensorNetwork(VirtualSensorNetwork):
    """Wrap a VSN and smooth position/orientation probabilities over time."""

    def __init__(self, base: VirtualSensorNetwork, alpha: float = 0.6):
        super().__init__(segmentation=base.segmentation, position=base.position, orientation=base.orientation)
        if not (0.0 < float(alpha) <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.base = base
        self.alpha = float(alpha)
        self._position_prob = None
        self._orientation_prob = None

    def reset_state(self) -> None:
        self._position_prob = None
        self._orientation_prob = None
        reset_base = getattr(self.base, "reset_state", None)
        if callable(reset_base):
            reset_base()

    def set_episode_seed(self, seed: int) -> None:
        setter = getattr(self.base, "set_episode_seed", None)
        if callable(setter):
            setter(seed)

    def forward(self, rgb=None, mask=None):  # noqa: D401 - inherited contract
        out = self.base(rgb=rgb, mask=mask)
        if self.alpha >= 1.0:
            return out

        if self._position_prob is None or self._position_prob.shape != out.position_prob.shape:
            position_prob = out.position_prob
        else:
            prev = self._position_prob.to(out.position_prob.device)
            position_prob = self.alpha * out.position_prob + (1.0 - self.alpha) * prev
            position_prob = position_prob / position_prob.flatten(1).sum(dim=1).clamp_min(1e-8).reshape(-1, 1, 1)

        if self._orientation_prob is None or self._orientation_prob.shape != out.orientation_prob.shape:
            orientation_prob = out.orientation_prob
        else:
            prev = self._orientation_prob.to(out.orientation_prob.device)
            orientation_prob = self.alpha * out.orientation_prob + (1.0 - self.alpha) * prev
            orientation_prob = orientation_prob / orientation_prob.sum(dim=1).clamp_min(1e-8).reshape(-1, 1)

        self._position_prob = position_prob.detach().cpu()
        self._orientation_prob = orientation_prob.detach().cpu()

        import torch

        flat = position_prob.flatten(1)
        idx = flat.argmax(dim=1)
        grid = position_prob.shape[-1]
        vals = [decode_position(int(i // grid), int(i % grid), grid_size=grid) for i in idx.detach().cpu()]
        dxy = torch.as_tensor(vals, dtype=out.dxy_m.dtype, device=out.dxy_m.device)

        angles = getattr(self.orientation, "angles", None)
        if angles is None:
            from ..constants import ORIENTATION_ANGLES_DEG

            angles = ORIENTATION_ANGLES_DEG
        angle_values = torch.as_tensor(list(angles), dtype=out.dyaw_deg.dtype, device=out.dyaw_deg.device)
        dyaw = angle_values[orientation_prob.argmax(dim=1)]

        return VSNOutput(
            out.mask_logits,
            out.mask,
            out.position_logits,
            position_prob,
            out.orientation_scores,
            orientation_prob,
            dxy,
            dyaw,
            flat.max(dim=1).values,
            orientation_prob.max(dim=1).values,
        )


class EnsembleVirtualSensorNetwork(VirtualSensorNetwork):
    """Average several VSN probability predictions for the same observation."""

    def __init__(self, base: VirtualSensorNetwork, samples: int = 3):
        super().__init__(segmentation=base.segmentation, position=base.position, orientation=base.orientation)
        if int(samples) < 1:
            raise ValueError("samples must be >= 1")
        self.base = base
        self.samples = int(samples)

    def reset_state(self) -> None:
        reset_base = getattr(self.base, "reset_state", None)
        if callable(reset_base):
            reset_base()

    def set_episode_seed(self, seed: int) -> None:
        setter = getattr(self.base, "set_episode_seed", None)
        if callable(setter):
            setter(seed)

    def forward(self, rgb=None, mask=None):  # noqa: D401 - inherited contract
        outs = [self.base(rgb=rgb, mask=mask) for _ in range(self.samples)]
        if self.samples == 1:
            return outs[0]
        position_prob = sum(o.position_prob for o in outs) / float(self.samples)
        orientation_prob = sum(o.orientation_prob for o in outs) / float(self.samples)
        return _replace_probabilities(outs[-1], position_prob, orientation_prob, self.orientation)


def _replace_probabilities(out: VSNOutput, position_prob, orientation_prob, orientation_module) -> VSNOutput:
    import torch

    flat = position_prob.flatten(1)
    idx = flat.argmax(dim=1)
    grid = position_prob.shape[-1]
    vals = [decode_position(int(i // grid), int(i % grid), grid_size=grid) for i in idx.detach().cpu()]
    dxy = torch.as_tensor(vals, dtype=out.dxy_m.dtype, device=out.dxy_m.device)

    angles = getattr(orientation_module, "angles", None)
    if angles is None:
        from ..constants import ORIENTATION_ANGLES_DEG

        angles = ORIENTATION_ANGLES_DEG
    angle_values = torch.as_tensor(list(angles), dtype=out.dyaw_deg.dtype, device=out.dyaw_deg.device)
    dyaw = angle_values[orientation_prob.argmax(dim=1)]
    return VSNOutput(
        out.mask_logits,
        out.mask,
        out.position_logits,
        position_prob,
        out.orientation_scores,
        orientation_prob,
        dxy,
        dyaw,
        flat.max(dim=1).values,
        orientation_prob.max(dim=1).values,
    )


def _apply_camera_effects(
    image: np.ndarray,
    config: DisturbanceConfig,
    rng: np.random.Generator,
    *,
    scale: float,
    output_dtype,
) -> np.ndarray:
    """Apply label-free camera effects to one channel-first RGB image."""

    if not _has_extended_camera_effects(config):
        return np.asarray(image).astype(output_dtype, copy=True)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError("camera disturbances require channel-first RGB with shape (3, H, W)")

    if config.blur_kernel_px > 1:
        array = _box_blur_chw(array, int(config.blur_kernel_px))
    if config.motion_blur_px > 1:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        array = _motion_blur_chw(array, int(config.motion_blur_px), angle)
    if not np.isclose(config.intrinsic_scale, 1.0):
        array = _scale_intrinsics_chw(array, float(config.intrinsic_scale))
    if config.crop_fraction > 0:
        array = _crop_and_resize_chw(array, float(config.crop_fraction), rng)

    normalized = np.clip(array / float(scale), 0.0, 1.0)
    if not np.isclose(config.gamma, 1.0):
        normalized = np.power(normalized, float(config.gamma))
    gains = np.asarray(config.white_balance_rgb, dtype=np.float32).reshape(3, 1, 1)
    normalized *= gains
    if config.lighting_gradient != 0:
        height, width = normalized.shape[-2:]
        yy, xx = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        normalized *= 1.0 + float(config.lighting_gradient) * gradient[None, ...]
    normalized = normalized * float(config.lighting_gain) + float(config.lighting_bias)
    result = np.clip(normalized, 0.0, 1.0) * float(scale)
    dtype = np.dtype(output_dtype)
    if np.issubdtype(dtype, np.integer):
        result = np.rint(result)
    return result.astype(dtype, copy=False)


def _has_extended_camera_effects(config: DisturbanceConfig) -> bool:
    return bool(
        config.blur_kernel_px > 1
        or config.motion_blur_px > 1
        or not np.isclose(config.gamma, 1.0)
        or any(not np.isclose(value, 1.0) for value in config.white_balance_rgb)
        or not np.isclose(config.intrinsic_scale, 1.0)
        or config.crop_fraction > 0
        or not np.isclose(config.lighting_gain, 1.0)
        or not np.isclose(config.lighting_bias, 0.0)
        or not np.isclose(config.lighting_gradient, 0.0)
    )


def _box_blur_chw(image: np.ndarray, kernel_px: int) -> np.ndarray:
    radius = max(1, int(kernel_px) // 2)
    total = np.zeros_like(image, dtype=np.float32)
    padded = np.pad(image, ((0, 0), (radius, radius), (radius, radius)), mode="edge")
    width = 2 * radius + 1
    for dy in range(width):
        for dx in range(width):
            total += padded[:, dy : dy + image.shape[1], dx : dx + image.shape[2]]
    return total / float(width * width)


def _motion_blur_chw(image: np.ndarray, length_px: int, angle: float) -> np.ndarray:
    half = max(1, int(length_px) // 2)
    samples: list[np.ndarray] = []
    for offset in range(-half, half + 1):
        dx = int(round(np.cos(angle) * offset))
        dy = int(round(np.sin(angle) * offset))
        samples.append(_shift_edge_chw(image, dx=dx, dy=dy))
    return np.mean(samples, axis=0, dtype=np.float32)


def _shift_edge_chw(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = image.shape[-2:]
    y = np.clip(np.arange(height) - int(dy), 0, height - 1)
    x = np.clip(np.arange(width) - int(dx), 0, width - 1)
    return image[:, y[:, None], x[None, :]]


def _resize_chw(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Dependency-free bilinear resize for small robustness-evaluation images."""

    source_h, source_w = image.shape[-2:]
    if (source_h, source_w) == (height, width):
        return image.copy()
    y = np.linspace(0.0, max(0, source_h - 1), max(1, int(height)), dtype=np.float32)
    x = np.linspace(0.0, max(0, source_w - 1), max(1, int(width)), dtype=np.float32)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_h - 1)
    x1 = np.minimum(x0 + 1, source_w - 1)
    wy = (y - y0).reshape(1, -1, 1)
    wx = (x - x0).reshape(1, 1, -1)
    top = image[:, y0[:, None], x0[None, :]] * (1.0 - wx) + image[:, y0[:, None], x1[None, :]] * wx
    bottom = image[:, y1[:, None], x0[None, :]] * (1.0 - wx) + image[:, y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def _scale_intrinsics_chw(image: np.ndarray, intrinsic_scale: float) -> np.ndarray:
    """Approximate focal-length scaling while preserving output resolution."""

    height, width = image.shape[-2:]
    scaled_h = max(1, int(round(height * intrinsic_scale)))
    scaled_w = max(1, int(round(width * intrinsic_scale)))
    scaled = _resize_chw(image, scaled_h, scaled_w)
    if intrinsic_scale >= 1.0:
        y0 = max(0, (scaled_h - height) // 2)
        x0 = max(0, (scaled_w - width) // 2)
        return scaled[:, y0 : y0 + height, x0 : x0 + width]
    canvas = np.zeros_like(image, dtype=np.float32)
    y0 = (height - scaled_h) // 2
    x0 = (width - scaled_w) // 2
    canvas[:, y0 : y0 + scaled_h, x0 : x0 + scaled_w] = scaled
    return canvas


def _crop_and_resize_chw(image: np.ndarray, crop_fraction: float, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[-2:]
    crop_h = max(1, int(round(height * (1.0 - crop_fraction))))
    crop_w = max(1, int(round(width * (1.0 - crop_fraction))))
    y0 = int(rng.integers(0, max(1, height - crop_h + 1)))
    x0 = int(rng.integers(0, max(1, width - crop_w + 1)))
    return _resize_chw(image[:, y0 : y0 + crop_h, x0 : x0 + crop_w], height, width)


def _shift_zero_fill(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift ``mask`` without wraparound; exposed for tests via module import."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out
