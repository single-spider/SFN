"""Deterministic, replayable image-space domain randomization.

Records stay at version 1 so existing schema-v2 datasets and validators remain
valid.  The added fields are an additive record-contract extension.  Geometry
is shared by RGB, mask, and optional depth; every other family changes sensor
data only, never semantic labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import cv2
import numpy as np

LEVELS = {"none", "light", "medium", "heavy"}
_SCALE = {"none": 0.0, "light": 0.35, "medium": 0.65, "heavy": 1.0}


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Additive family switches; defaults preserve the level-only API."""

    color_illumination_gradient: bool = True
    material_texture_background: bool = True
    shadow: bool = True
    shot_noise: bool = True
    motion_blur: bool = True
    gamma: bool = True
    white_balance: bool = True
    seam: bool = True
    depth_noise: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> DomainRandomizationConfig:
        if value is None:
            return cls()
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown domain-randomization config key(s): {', '.join(unknown)}")
        for name, enabled in value.items():
            if not isinstance(enabled, bool):
                raise ValueError(f"domain-randomization config {name} must be boolean")
        return cls(**dict(value))


def _config(value: DomainRandomizationConfig | Mapping[str, Any] | None) -> DomainRandomizationConfig:
    if isinstance(value, DomainRandomizationConfig):
        return value
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("config must be DomainRandomizationConfig or a mapping")
    return DomainRandomizationConfig.from_mapping(value)


def _seed(generator: np.random.Generator) -> int:
    return int(generator.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))


def _neutral_record(level: str, seed: int | None, cfg: DomainRandomizationConfig) -> dict[str, Any]:
    return {
        "version": 1,
        "level": level,
        "seed": seed,
        "config": asdict(cfg),
        "order": [
            "camera", "material_texture_background", "color_illumination_gradient", "shadow", "seam",
            "exposure", "white_balance", "gamma", "shot_noise", "noise", "motion_blur", "blur",
            "occlusion", "depth_noise",
        ],
        "camera": {"dx_px": 0.0, "dy_px": 0.0, "rotation_deg": 0.0, "scale": 1.0},
        "material_texture_background": {"enabled": False},
        "color_illumination_gradient": {"enabled": False},
        "shadow": {"enabled": False},
        "seam": {"enabled": False},
        "exposure": {"gain": 1.0, "bias": 0.0},
        "white_balance": {"enabled": False, "gains_rgb": [1.0, 1.0, 1.0]},
        "gamma": {"enabled": False, "value": 1.0},
        "shot_noise": {"enabled": False},
        "noise": {"sigma": 0.0},
        "motion_blur": {"enabled": False, "kernel": 1, "angle_deg": 0.0},
        "blur": {"kernel": 1},
        "occlusion": {"enabled": False},
        "depth_noise": {"enabled": False},
    }


def _sample_record(
    shape: tuple[int, int], level: str, seed: int | None, generator: np.random.Generator,
    cfg: DomainRandomizationConfig, *, has_depth: bool,
) -> dict[str, Any]:
    strength = _SCALE[level]
    record = _neutral_record(level, seed, cfg)
    if strength == 0:
        return record
    h, w = shape
    record["camera"] = {
        "dx_px": float(generator.uniform(-8, 8) * strength),
        "dy_px": float(generator.uniform(-6, 6) * strength),
        "rotation_deg": float(generator.uniform(-3, 3) * strength),
        "scale": float(1.0 + generator.uniform(-0.035, 0.035) * strength),
    }
    if cfg.material_texture_background:
        record["material_texture_background"] = {
            "enabled": True,
            "class_gains_rgb": [
                [float(generator.uniform(1 - 0.22 * strength, 1 + 0.22 * strength)) for _ in range(3)]
                for _ in range(3)
            ],
            "texture_amplitude": float(generator.uniform(0, 24) * strength),
            "texture_scale_px": int(generator.integers(10, 31)),
            "seed": _seed(generator),
        }
    if cfg.color_illumination_gradient:
        record["color_illumination_gradient"] = {
            "enabled": True,
            "angle_deg": float(generator.uniform(0, 360)),
            "amplitude": float(generator.uniform(-0.32, 0.32) * strength),
            "color_rgb": [float(generator.uniform(0.8, 1.2)) for _ in range(3)],
        }
    if cfg.shadow:
        record["shadow"] = {
            "enabled": bool(generator.random() < 0.75),
            "center_xy": [float(generator.uniform(0, w - 1)), float(generator.uniform(0, h - 1))],
            "axes_xy": [float(generator.uniform(0.18, 0.65) * w), float(generator.uniform(0.15, 0.55) * h)],
            "angle_deg": float(generator.uniform(0, 180)),
            "opacity": float(generator.uniform(0.08, 0.48) * strength),
            "softness_px": float(generator.uniform(3, 15)),
        }
    if cfg.seam:
        record["seam"] = {
            "enabled": True,
            "width_px": int(generator.integers(1, 4)),
            # Negative values reduce seam contrast; positive values enhance it.
            "contrast": float(generator.uniform(-0.55, 0.55) * strength),
        }
    record["exposure"] = {
        "gain": float(generator.uniform(1 - 0.3 * strength, 1 + 0.3 * strength)),
        "bias": float(generator.uniform(-28, 28) * strength),
    }
    if cfg.white_balance:
        record["white_balance"] = {
            "enabled": True,
            "gains_rgb": [float(generator.uniform(1 - 0.2 * strength, 1 + 0.2 * strength)) for _ in range(3)],
        }
    if cfg.gamma:
        record["gamma"] = {"enabled": True, "value": float(generator.uniform(0.72, 1.38) ** strength)}
    if cfg.shot_noise:
        record["shot_noise"] = {
            "enabled": True,
            "peak_electrons": float(generator.uniform(140, 900) / max(strength, 0.01)),
            "seed": _seed(generator),
        }
    record["noise"] = {"sigma": float(generator.uniform(0, 14) * strength), "seed": _seed(generator)}
    if cfg.motion_blur:
        max_length = 1 + 2 * int(round(4 * strength))
        record["motion_blur"] = {
            "enabled": max_length > 1,
            "kernel": int(generator.choice(np.arange(1, max_length + 1, 2))),
            "angle_deg": float(generator.uniform(0, 180)),
        }
    max_kernel = 1 + 2 * int(round(2 * strength))
    record["blur"] = {"kernel": int(generator.choice(np.arange(1, max_kernel + 1, 2)))}
    enabled = bool(generator.random() < 0.55 * strength)
    occlusion: dict[str, Any] = {"enabled": enabled}
    if enabled:
        ow = int(generator.integers(max(1, int(w * 0.04)), max(2, int(w * (0.04 + 0.16 * strength)))))
        oh = int(generator.integers(max(1, int(h * 0.04)), max(2, int(h * (0.04 + 0.16 * strength)))))
        occlusion.update(
            x=int(generator.integers(0, w - ow + 1)), y=int(generator.integers(0, h - oh + 1)),
            width=ow, height=oh, color=[int(v) for v in generator.integers(0, 256, size=3, dtype=np.uint8)],
        )
    record["occlusion"] = occlusion
    if cfg.depth_noise and has_depth:
        record["depth_noise"] = {
            "enabled": True,
            "sigma_m": float(generator.uniform(0.0001, 0.0015) * strength),
            "relative_sigma": float(generator.uniform(0.0005, 0.005) * strength),
            "dropout_probability": float(generator.uniform(0, 0.008) * strength),
            "seed": _seed(generator),
        }
    return record


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    radius = center
    angle = np.deg2rad(angle_deg)
    p1 = (int(round(center - radius * np.cos(angle))), int(round(center - radius * np.sin(angle))))
    p2 = (int(round(center + radius * np.cos(angle))), int(round(center + radius * np.sin(angle))))
    cv2.line(kernel, p1, p2, 1.0, 1)
    return kernel / max(float(kernel.sum()), 1.0)


def _apply_record(
    rgb: np.ndarray, mask: np.ndarray, record: Mapping[str, Any], depth: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    h, w = mask.shape
    camera = record["camera"]
    matrix = cv2.getRotationMatrix2D(
        ((w - 1) / 2, (h - 1) / 2), float(camera["rotation_deg"]), float(camera["scale"])
    )
    matrix[:, 2] = matrix[:, 2] + np.asarray(
        [float(camera["dx_px"]), float(camera["dy_px"])], dtype=matrix.dtype
    )
    hwc = np.moveaxis(rgb, 0, -1)
    work = cv2.warpAffine(hwc, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out_mask = cv2.warpAffine(
        mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(mask.dtype, copy=False)
    out_depth = None
    if depth is not None:
        out_depth = cv2.warpAffine(
            depth, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        ).astype(depth.dtype, copy=False)
    work = work.astype(np.float32)

    material = record.get("material_texture_background", {"enabled": False})
    if material.get("enabled"):
        gains = np.asarray(material["class_gains_rgb"], dtype=np.float32)
        for class_id in range(min(3, gains.shape[0])):
            work[out_mask == class_id] *= gains[class_id]
        scale = max(2, int(material["texture_scale_px"]))
        small_h, small_w = max(2, (h + scale - 1) // scale), max(2, (w + scale - 1) // scale)
        coarse_texture = np.random.default_rng(int(material["seed"])).normal(
            0, 1, size=(small_h, small_w)
        ).astype(np.float32)
        texture = np.asarray(cv2.resize(coarse_texture, (w, h), interpolation=cv2.INTER_CUBIC), dtype=np.float32)
        work = work + texture[..., None] * float(material["texture_amplitude"])

    gradient = record.get("color_illumination_gradient", {"enabled": False})
    if gradient.get("enabled"):
        yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
        angle = np.deg2rad(float(gradient["angle_deg"]))
        field = np.clip(xx * np.cos(angle) + yy * np.sin(angle), -1, 1)
        color = np.asarray(gradient["color_rgb"], dtype=np.float32)
        work *= 1.0 + field[..., None] * float(gradient["amplitude"]) * color

    shadow = record.get("shadow", {"enabled": False})
    if shadow.get("enabled"):
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        center = tuple(int(round(v)) for v in shadow["center_xy"])
        axes = tuple(max(1, int(round(v))) for v in shadow["axes_xy"])
        cv2.ellipse(shadow_mask, center, axes, float(shadow["angle_deg"]), 0, 360, 1.0, -1)
        sigma = float(shadow["softness_px"])
        soft_shadow = np.asarray(
            cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=sigma, sigmaY=sigma), dtype=np.float32
        )
        work = work * (1.0 - soft_shadow[..., None] * float(shadow["opacity"]))

    seam = record.get("seam", {"enabled": False})
    if seam.get("enabled"):
        width = max(1, int(seam["width_px"]))
        seam_kernel = np.ones((2 * width + 1, 2 * width + 1), dtype=np.uint8)
        edge = cv2.morphologyEx(out_mask.astype(np.uint16), cv2.MORPH_GRADIENT, seam_kernel) > 0
        contrast = float(seam["contrast"])
        if contrast >= 0:
            work[edge] = (work[edge] - 127.5) * (1.0 + contrast) + 127.5
        else:
            local = cv2.GaussianBlur(work, (0, 0), sigmaX=max(1.0, width))
            work[edge] = work[edge] * (1.0 + contrast) + local[edge] * (-contrast)

    exposure = record["exposure"]
    work = work * float(exposure["gain"]) + float(exposure["bias"])
    wb = record.get("white_balance", {"enabled": False})
    if wb.get("enabled"):
        work *= np.asarray(wb["gains_rgb"], dtype=np.float32)
    gamma = record.get("gamma", {"enabled": False})
    if gamma.get("enabled"):
        work = 255.0 * np.power(np.clip(work, 0, 255) / 255.0, float(gamma["value"]))
    shot = record.get("shot_noise", {"enabled": False})
    if shot.get("enabled"):
        peak = float(shot["peak_electrons"])
        lam = np.clip(work, 0, 255) * (peak / 255.0)
        work = np.random.default_rng(int(shot["seed"])).poisson(lam).astype(np.float32) * (255.0 / peak)
    noise = record["noise"]
    sigma = float(noise["sigma"])
    if sigma:
        # Legacy records had no noise seed and can still replay when their root seed exists.
        noise_seed = noise.get("seed", record.get("seed"))
        if noise_seed is None:
            raise ValueError("record lacks the seed required to replay Gaussian noise")
        work += np.random.default_rng(int(noise_seed)).normal(0, sigma, work.shape).astype(np.float32)
    motion = record.get("motion_blur", {"enabled": False})
    if motion.get("enabled") and int(motion["kernel"]) > 1:
        work = cv2.filter2D(work, -1, _motion_kernel(int(motion["kernel"]), float(motion["angle_deg"])))
    work = np.clip(work, 0, 255).astype(np.uint8)
    blur_kernel = int(record["blur"]["kernel"])
    if blur_kernel > 1:
        work = cv2.GaussianBlur(work, (blur_kernel, blur_kernel), 0)
    occlusion = record["occlusion"]
    if occlusion.get("enabled"):
        x, y = int(occlusion["x"]), int(occlusion["y"])
        work[y : y + int(occlusion["height"]), x : x + int(occlusion["width"])] = occlusion["color"]

    depth_record = record.get("depth_noise", {"enabled": False})
    if out_depth is not None and depth_record.get("enabled"):
        assert depth is not None
        valid = out_depth > 0
        depth_rng = np.random.default_rng(int(depth_record["seed"]))
        sigma_depth = float(depth_record["sigma_m"]) + out_depth.astype(np.float32) * float(
            depth_record["relative_sigma"]
        )
        noisy = out_depth.astype(np.float32) + depth_rng.normal(0, 1, size=out_depth.shape).astype(
            np.float32
        ) * sigma_depth
        dropout = depth_rng.random(out_depth.shape) < float(depth_record["dropout_probability"])
        noisy[~valid | dropout] = 0.0
        out_depth = np.maximum(noisy, 0).astype(depth.dtype)
    return (
        np.ascontiguousarray(np.moveaxis(work, -1, 0)),
        np.ascontiguousarray(out_mask),
        None if out_depth is None else np.ascontiguousarray(out_depth),
    )


def replay_domain_randomization(
    rgb: np.ndarray, mask: np.ndarray, record: Mapping[str, Any], *, depth: np.ndarray | None = None,
):
    """Replay an augmentation record without consulting global or caller RNG state."""
    _validate_inputs(rgb, mask, depth)
    if record.get("version") != 1 or record.get("level") not in LEVELS:
        raise ValueError("unsupported domain-randomization record")
    out_rgb, out_mask, out_depth = _apply_record(rgb, mask, record, depth)
    return (out_rgb, out_mask) if depth is None else (out_rgb, out_mask, out_depth)


def _validate_inputs(rgb: np.ndarray, mask: np.ndarray, depth: np.ndarray | None) -> None:
    if rgb.ndim != 3 or rgb.shape[0] != 3 or mask.shape != rgb.shape[1:]:
        raise ValueError(f"expected rgb (3,H,W) and mask (H,W), got {rgb.shape} and {mask.shape}")
    if depth is not None:
        if depth.shape != mask.shape or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError("depth must be a floating HW array aligned with mask")
        if not np.all(np.isfinite(depth)) or np.any(depth < 0):
            raise ValueError("depth must contain finite non-negative metric values")


def apply_domain_randomization(
    rgb: np.ndarray,
    mask: np.ndarray,
    level: str = "none",
    rng: np.random.Generator | None = None,
    *,
    seed: int | None = None,
    return_record: bool = False,
    depth: np.ndarray | None = None,
    config: DomainRandomizationConfig | Mapping[str, Any] | None = None,
):
    """Sample and apply domain randomization to CHW RGB and optional HW depth.

    The historical RGB/mask result arity is unchanged.  Supplying depth adds an
    aligned depth result immediately before the optional record.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    _validate_inputs(rgb, mask, depth)
    cfg = _config(config)
    generator = rng if rng is not None else np.random.default_rng(seed)
    record = _sample_record(mask.shape, level, seed, generator, cfg, has_depth=depth is not None)
    out_rgb, out_mask, out_depth = _apply_record(rgb, mask, record, depth)
    values: tuple[Any, ...] = (out_rgb, out_mask) if depth is None else (out_rgb, out_mask, out_depth)
    return (*values, record) if return_record else values
