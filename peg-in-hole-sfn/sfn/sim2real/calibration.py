"""Portable, strictly validated camera-calibration records.

The JSON representation deliberately avoids OpenCV/YAML conventions whose
matrix ordering and transform direction are often implicit.  A transform in
this module always maps points from ``source_frame`` to ``target_frame``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA_VERSION = "sfn.camera_calibration/v1"
_DISTORTION_LENGTHS = {"none": {0}, "plumb_bob": {4, 5}, "rational_polynomial": {8}, "fisheye": {4}}

# Standard JSON Schema counterpart to ``CameraCalibration.from_dict``.  It is
# dependency-free metadata: callers may feed it to any Draft 2020-12 validator.
CAMERA_CALIBRATION_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://sfn.local/schemas/camera-calibration-v1.json",
    "title": "SFN camera calibration",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "camera_name", "units", "intrinsics", "distortion", "extrinsics"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "camera_name": {"type": "string", "minLength": 1},
        "units": {
            "type": "object",
            "additionalProperties": False,
            "required": ["length", "angle", "image"],
            "properties": {
                "length": {"const": "metre"},
                "angle": {"const": "radian"},
                "image": {"const": "pixel"},
            },
        },
        "intrinsics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["image_width_px", "image_height_px", "fx_px", "fy_px", "cx_px", "cy_px"],
            "properties": {
                "image_width_px": {"type": "integer", "minimum": 1},
                "image_height_px": {"type": "integer", "minimum": 1},
                "fx_px": {"type": "number", "exclusiveMinimum": 0},
                "fy_px": {"type": "number", "exclusiveMinimum": 0},
                "cx_px": {"type": "number"},
                "cy_px": {"type": "number"},
            },
        },
        "distortion": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["model", "coefficients"],
                    "properties": {"model": {"const": "none"}, "coefficients": {"type": "array", "maxItems": 0}},
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["model", "coefficients"],
                    "properties": {
                        "model": {"const": "plumb_bob"},
                        "coefficients": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 5},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["model", "coefficients"],
                    "properties": {
                        "model": {"const": "rational_polynomial"},
                        "coefficients": {"type": "array", "items": {"type": "number"}, "minItems": 8, "maxItems": 8},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["model", "coefficients"],
                    "properties": {
                        "model": {"const": "fisheye"},
                        "coefficients": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                    },
                },
            ]
        },
        "extrinsics": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_frame", "target_frame", "direction", "translation", "rotation_xyzw"],
            "properties": {
                "source_frame": {"type": "string", "minLength": 1},
                "target_frame": {"type": "string", "minLength": 1},
                "direction": {"const": "source_to_target"},
                "translation": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "rotation_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            },
        },
    },
}


def _strict_keys(value: Mapping[str, Any], required: set[str], where: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ValueError(f"{where}: " + ", ".join(details))


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def _vector(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must be a JSON array of length {length}")
    return tuple(_number(item, f"{where}[{index}]") for index, item in enumerate(value))


def _name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{where} must be a non-empty, trimmed string")
    return value


@dataclass(frozen=True)
class CameraIntrinsics:
    image_width_px: int
    image_height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.image_width_px, bool)
            or not isinstance(self.image_width_px, int)
            or self.image_width_px <= 0
        ):
            raise ValueError("image_width_px must be a positive integer")
        if (
            isinstance(self.image_height_px, bool)
            or not isinstance(self.image_height_px, int)
            or self.image_height_px <= 0
        ):
            raise ValueError("image_height_px must be a positive integer")
        for field_name in ("fx_px", "fy_px", "cx_px", "cy_px"):
            value = _number(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("focal lengths must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray([[self.fx_px, 0.0, self.cx_px], [0.0, self.fy_px, self.cy_px], [0.0, 0.0, 1.0]])

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "fx_px": self.fx_px,
            "fy_px": self.fy_px,
            "cx_px": self.cx_px,
            "cy_px": self.cy_px,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraIntrinsics:
        required = {"image_width_px", "image_height_px", "fx_px", "fy_px", "cx_px", "cy_px"}
        _strict_keys(value, required, "intrinsics")
        width, height = value["image_width_px"], value["image_height_px"]
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
        ):
            raise ValueError("intrinsics image dimensions must be integers")
        return cls(width, height, *(_number(value[k], f"intrinsics.{k}") for k in ("fx_px", "fy_px", "cx_px", "cy_px")))


@dataclass(frozen=True)
class CameraDistortion:
    model: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.model not in _DISTORTION_LENGTHS:
            raise ValueError(f"unsupported distortion model {self.model!r}")
        coefficients = tuple(_number(v, "distortion coefficient") for v in self.coefficients)
        object.__setattr__(self, "coefficients", coefficients)
        if len(coefficients) not in _DISTORTION_LENGTHS[self.model]:
            raise ValueError(f"{self.model} distortion requires {sorted(_DISTORTION_LENGTHS[self.model])} coefficients")

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "coefficients": list(self.coefficients)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraDistortion:
        _strict_keys(value, {"model", "coefficients"}, "distortion")
        if not isinstance(value["model"], str) or not isinstance(value["coefficients"], list):
            raise ValueError("distortion model must be a string and coefficients must be an array")
        return cls(value["model"], tuple(_number(v, "distortion.coefficients") for v in value["coefficients"]))


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform mapping ``source_frame`` coordinates to ``target_frame``."""

    source_frame: str
    target_frame: str
    direction: str
    translation: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_frame", _name(self.source_frame, "source_frame"))
        object.__setattr__(self, "target_frame", _name(self.target_frame, "target_frame"))
        if self.source_frame == self.target_frame:
            raise ValueError("source_frame and target_frame must differ")
        if self.direction != "source_to_target":
            raise ValueError("direction must be exactly 'source_to_target'")
        translation = tuple(_number(v, "translation") for v in self.translation)
        quaternion = np.asarray(tuple(_number(v, "rotation_xyzw") for v in self.rotation_xyzw), dtype=np.float64)
        if len(translation) != 3 or quaternion.shape != (4,):
            raise ValueError("translation and rotation_xyzw must have lengths 3 and 4")
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(f"rotation_xyzw must be a unit quaternion (norm={norm:.9g})")
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "rotation_xyzw", tuple(float(v) for v in quaternion))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "direction": self.direction,
            "translation": list(self.translation),
            "rotation_xyzw": list(self.rotation_xyzw),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraExtrinsics:
        required = {"source_frame", "target_frame", "direction", "translation", "rotation_xyzw"}
        _strict_keys(value, required, "extrinsics")
        return cls(
            _name(value["source_frame"], "extrinsics.source_frame"),
            _name(value["target_frame"], "extrinsics.target_frame"),
            _name(value["direction"], "extrinsics.direction"),
            _vector(value["translation"], 3, "extrinsics.translation"),  # type: ignore[arg-type]
            _vector(value["rotation_xyzw"], 4, "extrinsics.rotation_xyzw"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CalibrationUnits:
    length: str = "metre"
    angle: str = "radian"
    image: str = "pixel"

    def __post_init__(self) -> None:
        if (self.length, self.angle, self.image) != ("metre", "radian", "pixel"):
            raise ValueError("units must be exactly metre, radian, and pixel")

    def to_dict(self) -> dict[str, str]:
        return {"length": self.length, "angle": self.angle, "image": self.image}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationUnits:
        _strict_keys(value, {"length", "angle", "image"}, "units")
        return cls(value["length"], value["angle"], value["image"])


@dataclass(frozen=True)
class CameraCalibration:
    camera_name: str
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    extrinsics: CameraExtrinsics
    units: CalibrationUnits = CalibrationUnits()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_name", _name(self.camera_name, "camera_name"))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.extrinsics.source_frame != self.camera_name:
            raise ValueError("extrinsics.source_frame must equal camera_name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_name": self.camera_name,
            "units": self.units.to_dict(),
            "intrinsics": self.intrinsics.to_dict(),
            "distortion": self.distortion.to_dict(),
            "extrinsics": self.extrinsics.to_dict(),
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False) + "\n"
        if path is not None:
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CameraCalibration:
        if not isinstance(value, Mapping):
            raise ValueError("calibration root must be a JSON object")
        required = {"schema_version", "camera_name", "units", "intrinsics", "distortion", "extrinsics"}
        _strict_keys(value, required, "calibration")
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value['schema_version']!r}")
        for key in ("units", "intrinsics", "distortion", "extrinsics"):
            if not isinstance(value[key], Mapping):
                raise ValueError(f"{key} must be a JSON object")
        return cls(
            camera_name=_name(value["camera_name"], "camera_name"),
            units=CalibrationUnits.from_dict(value["units"]),
            intrinsics=CameraIntrinsics.from_dict(value["intrinsics"]),
            distortion=CameraDistortion.from_dict(value["distortion"]),
            extrinsics=CameraExtrinsics.from_dict(value["extrinsics"]),
            schema_version=value["schema_version"],
        )

    @classmethod
    def from_json(cls, source: str | Path) -> CameraCalibration:
        """Read a path, or parse a JSON string when no such path exists."""
        if isinstance(source, Path) or (isinstance(source, str) and not source.lstrip().startswith("{")):
            path = Path(source)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid calibration JSON in {path}: {exc}") from exc
        else:
            try:
                payload = json.loads(str(source))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid calibration JSON: {exc}") from exc
        return cls.from_dict(payload)

    def undistort(self, image_rgb: np.ndarray) -> np.ndarray:
        """Undistort one RGB frame, requiring its calibrated resolution."""
        image = np.asarray(image_rgb)
        expected = (self.intrinsics.image_height_px, self.intrinsics.image_width_px)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"frame must have HxWx3 shape, got {image.shape}")
        if image.shape[:2] != expected:
            raise ValueError(
                f"frame resolution {image.shape[1]}x{image.shape[0]} does not match calibration {expected[1]}x{expected[0]}"
            )
        if self.distortion.model == "none" or not any(self.distortion.coefficients):
            return image.copy()
        camera_matrix = self.intrinsics.matrix
        coefficients = np.asarray(self.distortion.coefficients, dtype=np.float64)
        if self.distortion.model == "fisheye":
            return cv2.fisheye.undistortImage(image, camera_matrix, coefficients.reshape(-1, 1), Knew=camera_matrix)
        return cv2.undistort(image, camera_matrix, coefficients)


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    return CameraCalibration.from_json(Path(path))


def save_camera_calibration(calibration: CameraCalibration, path: str | Path) -> Path:
    output = Path(path)
    calibration.to_json(output)
    return output
