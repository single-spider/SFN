from __future__ import annotations

from sfn.panda.validation import validate_attachment, validate_command_tracking, validate_ik_grid


def test_panda_attachment_smoke_passes():
    _, summary = validate_attachment("square-concave1", steps=10)
    assert summary["success"]


def test_panda_cardinal_command_tracking_passes():
    _, summary = validate_command_tracking("square-concave1", trials=6)
    assert summary["success"]
    assert summary["cardinal_signs_ok"]


def test_panda_ik_grid_smoke_passes():
    _, summary = validate_ik_grid("square-concave1", grid_mm=[-1, 0, 1], grid_yaw_deg=[-1, 0, 1])
    assert summary["success"]
