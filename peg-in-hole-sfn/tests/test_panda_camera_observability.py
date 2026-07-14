from __future__ import annotations

import numpy as np
import pytest
from sfn.panda.camera_observability import (
    CameraCandidate,
    SweepThresholds,
    canonical_angle_deg,
    decode_native_segmentation,
    infer_rotational_symmetry_order,
    sweep_camera_observability,
    symmetry_aware_yaw_distance_deg,
    validate_native_mask,
)
from sfn.panda.config import PandaConfig


def test_decode_and_validate_native_body_id_mask():
    peg_id = 17
    seam_id = 29
    # High bits contain linkIndex + 1 in PyBullet's packed representation.
    segmentation = np.asarray(
        [
            [-1, seam_id, seam_id],
            [peg_id | (3 << 24), peg_id, 777],
            [-1, -1, -1],
        ],
        dtype=np.int64,
    )
    mask = decode_native_segmentation(segmentation, peg_id, seam_id)
    assert mask.tolist() == [[0, 2, 2], [1, 1, 0], [0, 0, 0]]
    validation = validate_native_mask(mask, expected_shape=(3, 3), min_peg_pixels=2, min_seam_pixels=2)
    assert validation.valid
    assert validation.peg_pixels == 2
    assert validation.seam_pixels == 2
    assert validation.peg_clipped
    assert validation.seam_clipped


def test_native_mask_validator_reports_missing_classes_and_bad_labels():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[2, 2] = 9
    validation = validate_native_mask(mask, min_peg_pixels=1, min_seam_pixels=1)
    assert not validation.valid
    assert any("unexpected labels" in error for error in validation.errors)
    assert any("peg pixels" in error for error in validation.errors)
    assert any("seam pixels" in error for error in validation.errors)


def test_symmetry_aware_angles_and_silhouette_inference():
    assert canonical_angle_deg(46.0, 90.0) == -44.0
    assert symmetry_aware_yaw_distance_deg(1.0, 89.0, 90.0) == 2.0

    square = np.zeros((101, 101), dtype=np.uint8)
    square[30:71, 30:71] = 1
    order, scores = infer_rotational_symmetry_order(square, max_order=8, iou_threshold=0.95)
    assert order == 4
    assert scores[4] >= 0.95

    rectangle = np.zeros((101, 101), dtype=np.uint8)
    rectangle[38:63, 25:76] = 1
    order, _scores = infer_rotational_symmetry_order(rectangle, max_order=8, iou_threshold=0.95)
    assert order == 2


def test_actual_panda_camera_sweep_records_visibility_sensitivity_and_yaw():
    pytest.importorskip("pybullet")
    candidate = CameraCandidate(
        eye_offset_m=(0.0, 0.0, 0.20),
        target_offset_m=(0.0, 0.0, 0.03),
        fov_y_deg=45.0,
        width=250,
        height=200,
        far=0.5,
        name="known_visible",
    )
    poses = [
        (-0.005, 0.0, 0.0),
        (0.005, 0.0, 0.0),
        (0.0, -0.005, 0.0),
        (0.0, 0.005, 0.0),
        (0.0, 0.0, -8.0),
        (0.0, 0.0, 8.0),
    ]
    report = sweep_camera_observability(
        shapes=["square-square"],
        poses=poses,
        candidates=[candidate],
        panda_config=PandaConfig(
            native_camera=True,
            mesh_derived_alignment_z=True,
            use_convex_decomposition=False,
            command_steps=1,
        ),
        thresholds=SweepThresholds(
            min_peg_pixels=5,
            min_seam_pixels=2,
            min_visible_fraction=0.6,
            min_unclipped_fraction=0.8,
            min_xy_sensitivity_px_per_mm=0.01,
            min_yaw_change_per_deg=0.0,
        ),
    )
    summary = report["candidates"][0]
    assert summary["frame_count"] == len(poses)
    assert summary["visible_fraction"] >= 0.6
    assert summary["xy_sensitivity"]["minimum_px_per_mm"] > 0.01
    yaw = summary["yaw_diagnostics"]["square-square"]
    assert yaw["symmetry_order"] >= 2
    assert yaw["pair_count"] >= 1
    assert not report["core_integration"]["required"]
    assert report["core_integration"]["runtime_camera_switching_requires_core_change"]
