"""Calibration-aware offline ingestion and VSN inference for recordings."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .calibration import CameraCalibration

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ReplayFrame:
    image_rgb: np.ndarray
    index: int
    timestamp_s: float
    source: str


@dataclass(frozen=True)
class ReplayResult:
    """One portable replay result. Pose is the SFN ``dx, dy, dyaw`` contract."""

    index: int
    timestamp_s: float
    source: str
    width: int
    height: int
    calibrated: bool
    camera_frame: str | None
    reference_frame: str | None
    pose_dx_m: float | None
    pose_dy_m: float | None
    pose_dyaw_deg: float | None
    position_confidence: float | None
    orientation_confidence: float | None
    confidence: float | None
    valid: bool
    invalid_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        pose = None
        if self.pose_dx_m is not None:
            pose = {"dx_m": self.pose_dx_m, "dy_m": self.pose_dy_m, "dyaw_deg": self.pose_dyaw_deg}
        return {
            "index": self.index,
            "timestamp_s": self.timestamp_s,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "calibrated": self.calibrated,
            "camera_frame": self.camera_frame,
            "reference_frame": self.reference_frame,
            "pose": pose,
            "position_confidence": self.position_confidence,
            "orientation_confidence": self.orientation_confidence,
            "confidence": self.confidence,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def iter_image_folder(path: str | Path, *, fps: float = 30.0) -> Iterator[ReplayFrame]:
    root = Path(path)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not root.is_dir():
        raise FileNotFoundError(f"image folder does not exist: {root}")
    files = sorted((p for p in root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)
    if not files:
        raise ValueError(f"no supported images found in {root}")
    for index, image_path in enumerate(files):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not decode image: {image_path}")
        yield ReplayFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), index, index / fps, str(image_path))


def iter_video(path: str | Path) -> Iterator[ReplayFrame]:
    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_s = timestamp_ms / 1000.0 if np.isfinite(timestamp_ms) and timestamp_ms > 0 else index / fps
            yield ReplayFrame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), index, timestamp_s, str(video_path))
            index += 1
    finally:
        capture.release()


def iter_replay(path: str | Path, *, image_fps: float = 30.0) -> Iterator[ReplayFrame]:
    source = Path(path)
    return iter_image_folder(source, fps=image_fps) if source.is_dir() else iter_video(source)


def _scalar(value: Any, name: str) -> float:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.size != 1:
        raise ValueError(f"VSN {name} must contain one value per replay frame, got shape {array.shape}")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"VSN {name} is not finite")
    return result


def _dxy(value: Any) -> tuple[float, float]:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.size != 2:
        raise ValueError(f"VSN dxy_m must contain two values per replay frame, got shape {array.shape}")
    dx, dy = (float(v) for v in array.reshape(-1))
    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError("VSN dxy_m is not finite")
    return dx, dy


def replay_frames(
    frames: Iterable[ReplayFrame],
    *,
    vsn: Any | None = None,
    calibration: CameraCalibration | None = None,
    device: str | torch.device = "cpu",
    min_position_confidence: float = 0.0,
    min_orientation_confidence: float = 0.0,
    max_frames: int | None = None,
) -> Iterator[ReplayResult]:
    """Undistort frames and optionally infer pose with a VSN-like callable.

    ``valid`` means both confidence gates passed.  Without a VSN, metadata is
    still emitted but pose/confidence are null and validity is false.
    """
    for name, threshold in (
        ("min_position_confidence", min_position_confidence),
        ("min_orientation_confidence", min_orientation_confidence),
    ):
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative or None")
    if vsn is not None:
        to_method = getattr(vsn, "to", None)
        if callable(to_method):
            vsn = to_method(device)
        eval_method = getattr(vsn, "eval", None)
        if callable(eval_method):
            eval_method()

    for count, frame in enumerate(frames):
        if max_frames is not None and count >= max_frames:
            break
        image = calibration.undistort(frame.image_rgb) if calibration is not None else np.asarray(frame.image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"replay frame {frame.index} must be HxWx3 RGB")
        height, width = image.shape[:2]
        camera_frame = calibration.extrinsics.source_frame if calibration else None
        reference_frame = calibration.extrinsics.target_frame if calibration else None
        if vsn is None:
            yield ReplayResult(
                frame.index,
                frame.timestamp_s,
                frame.source,
                width,
                height,
                calibration is not None,
                camera_frame,
                reference_frame,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
                "vsn_not_configured",
            )
            continue

        # VirtualSensorNetwork accepts uint8 NCHW RGB and performs its own normalization.
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = vsn(rgb=tensor)
        dx, dy = _dxy(output.dxy_m)
        dyaw = _scalar(output.dyaw_deg, "dyaw_deg")
        position_confidence = _scalar(output.position_confidence, "position_confidence")
        orientation_confidence = _scalar(output.orientation_confidence, "orientation_confidence")
        confidence = min(position_confidence, orientation_confidence)
        reasons = []
        if position_confidence < min_position_confidence:
            reasons.append("position_confidence_below_threshold")
        if orientation_confidence < min_orientation_confidence:
            reasons.append("orientation_confidence_below_threshold")
        yield ReplayResult(
            frame.index,
            frame.timestamp_s,
            frame.source,
            width,
            height,
            calibration is not None,
            camera_frame,
            reference_frame,
            dx,
            dy,
            dyaw,
            position_confidence,
            orientation_confidence,
            confidence,
            not reasons,
            ";".join(reasons) or None,
        )


def replay_to_jsonl(
    source: str | Path,
    output: str | Path,
    *,
    vsn: Any | None = None,
    calibration: CameraCalibration | None = None,
    image_fps: float = 30.0,
    device: str | torch.device = "cpu",
    min_position_confidence: float = 0.0,
    min_orientation_confidence: float = 0.0,
    max_frames: int | None = None,
) -> list[ReplayResult]:
    """Replay ``source`` and atomically replace a JSONL result manifest."""
    results = list(
        replay_frames(
            iter_replay(source, image_fps=image_fps),
            vsn=vsn,
            calibration=calibration,
            device=device,
            min_position_confidence=min_position_confidence,
            min_orientation_confidence=min_orientation_confidence,
            max_frames=max_frames,
        )
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row.to_dict(), allow_nan=False) + "\n" for row in results), encoding="utf-8"
    )
    temporary.replace(destination)
    return results
