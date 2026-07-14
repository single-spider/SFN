from __future__ import annotations

import json

import pytest
from sfn.constants import ALL_EXPECTED_SHAPES
from sfn.showcase.replay import load_replay, replay_fingerprint
from sfn.showcase.schema import REPLAY_SCHEMA_ID, ReplayDocument


def _frame(index: int) -> dict:
    return {
        "frame": index,
        "phase": "ready",
        "joint_positions": [0.0] * 9,
        "peg_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "hole_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "xy_error_mm": 1.0,
        "yaw_error_deg": 1.0,
    }


def test_replay_loader_and_fingerprint_are_deterministic(tmp_path) -> None:
    path = tmp_path / "replay.json"
    path.write_text(
        json.dumps(
            {
                "schema": REPLAY_SCHEMA_ID,
                "session": {"shape": ALL_EXPECTED_SHAPES[0], "method": "sfms", "seed": 3},
                "frames": [_frame(0), _frame(1)],
            }
        ),
        encoding="utf-8",
    )
    first = load_replay(path)
    assert first == load_replay(path)
    assert replay_fingerprint(first) == replay_fingerprint(first)


def test_replay_requires_contiguous_frames() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        ReplayDocument.model_validate(
            {
                "session": {"shape": ALL_EXPECTED_SHAPES[0]},
                "frames": [_frame(1)],
            }
        )
