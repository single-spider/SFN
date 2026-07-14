from __future__ import annotations

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from sfn.sim2real.calibration import (
    CalibrationUnits,
    CameraCalibration,
    CameraDistortion,
    CameraExtrinsics,
    CameraIntrinsics,
    load_camera_calibration,
)
from sfn.sim2real.replay import ReplayFrame, iter_video, replay_frames, replay_to_jsonl


def _calibration(width: int = 8, height: int = 6) -> CameraCalibration:
    return CameraCalibration(
        camera_name="camera_optical",
        intrinsics=CameraIntrinsics(width, height, 100.0, 101.0, width / 2, height / 2),
        distortion=CameraDistortion("plumb_bob", (0.0, 0.0, 0.0, 0.0, 0.0)),
        extrinsics=CameraExtrinsics(
            "camera_optical", "task", "source_to_target", (0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)
        ),
        units=CalibrationUnits(),
    )


def test_calibration_json_round_trip_is_strict(tmp_path):
    calibration = _calibration()
    path = tmp_path / "camera.json"
    calibration.to_json(path)
    assert load_camera_calibration(path) == calibration
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["intrinsics"]["surprise"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        CameraCalibration.from_dict(payload)
    payload["intrinsics"].pop("surprise")
    payload["extrinsics"]["direction"] = "target_to_source"
    with pytest.raises(ValueError, match="source_to_target"):
        CameraCalibration.from_dict(payload)


def test_calibration_rejects_wrong_resolution_and_non_unit_rotation():
    with pytest.raises(ValueError, match="unit quaternion"):
        CameraExtrinsics("camera", "task", "source_to_target", (0, 0, 0), (0, 0, 0, 2))
    with pytest.raises(ValueError, match="does not match calibration"):
        _calibration().undistort(np.zeros((5, 8, 3), dtype=np.uint8))


class _DummyVSN:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, *, rgb):
        assert rgb.shape == (1, 3, 6, 8)
        return SimpleNamespace(
            dxy_m=torch.tensor([[0.001, -0.002]]),
            dyaw_deg=torch.tensor([3.0]),
            position_confidence=torch.tensor([0.9]),
            orientation_confidence=torch.tensor([0.6]),
        )


def test_calibration_aware_replay_writes_pose_confidence_and_validity(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    bgr = np.zeros((6, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "000.png"), bgr)
    output = tmp_path / "replay.jsonl"
    results = replay_to_jsonl(
        image_dir,
        output,
        vsn=_DummyVSN(),
        calibration=_calibration(),
        min_position_confidence=0.8,
        min_orientation_confidence=0.7,
    )
    assert len(results) == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["calibrated"] is True
    assert row["camera_frame"] == "camera_optical"
    assert row["reference_frame"] == "task"
    assert row["pose"] == pytest.approx({"dx_m": 0.001, "dy_m": -0.002, "dyaw_deg": 3.0})
    assert row["confidence"] == pytest.approx(0.6)
    assert row["valid"] is False
    assert row["invalid_reason"] == "orientation_confidence_below_threshold"


def test_replay_without_vsn_has_explicit_null_invalid_output():
    frame = ReplayFrame(np.zeros((6, 8, 3), dtype=np.uint8), 0, 0.0, "memory")
    result = next(replay_frames([frame], calibration=_calibration()))
    assert result.pose_dx_m is None
    assert result.confidence is None
    assert not result.valid
    assert result.invalid_reason == "vsn_not_configured"


def test_video_replay_when_codec_is_available(tmp_path):
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (8, 6))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    try:
        for value in (10, 30):
            writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    frames = list(iter_video(path))
    if not frames:
        pytest.skip("OpenCV build cannot decode its MJPG output")
    assert [frame.index for frame in frames] == [0, 1]
    assert all(frame.image_rgb.shape == (6, 8, 3) for frame in frames)
