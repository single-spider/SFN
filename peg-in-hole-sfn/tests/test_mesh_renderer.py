from __future__ import annotations

import hashlib

import numpy as np
from sfn.config import CameraConfig
from sfn.envs.asset_registry import AssetRegistry
from sfn.envs.renderer import MeshOrthographicRenderer


def _centroid(mask: np.ndarray, label: int) -> tuple[float, float]:
    yy, xx = np.nonzero(mask == label)
    assert len(xx) > 0
    return float(xx.mean()), float(yy.mean())


def test_mesh_renderer_uses_every_actual_shape_and_produces_seam():
    registry = AssetRegistry()
    renderer = MeshOrthographicRenderer(
        CameraConfig(renderer_backend="mesh_orthographic", orthographic_pixels_per_mm=4.0),
        registry,
    )
    hashes = {}
    for shape in registry.list_shapes():
        out = renderer.render(np.zeros(3, dtype=np.float32), shape)
        assert out.rgb.shape == (3, 200, 250)
        assert out.mask.shape == (200, 250)
        assert set(np.unique(out.mask)) == {0, 1, 2}
        assert out.metadata is not None
        assert out.metadata["asset_faithful"] is True
        assert out.metadata["peg_pixels"] > 0
        assert out.metadata["seam_pixels"] > 0
        hashes[shape] = hashlib.sha256(out.mask.tobytes()).hexdigest()
    # At the configured resolution every supplied asset has a distinct nominal
    # silhouette/mask. This guards against returning to shape-name rectangles.
    assert len(set(hashes.values())) == len(hashes)


def test_mesh_renderer_pose_signs_match_sfn_contract():
    renderer = MeshOrthographicRenderer(
        CameraConfig(renderer_backend="mesh_orthographic", orthographic_pixels_per_mm=4.0)
    )
    shape = "square-triangle"
    center = renderer.render(np.asarray([0.0, 0.0, 0.0]), shape).mask
    plus_x = renderer.render(np.asarray([0.001, 0.0, 0.0]), shape).mask
    plus_y = renderer.render(np.asarray([0.0, 0.001, 0.0]), shape).mask
    cx, cy = _centroid(center, 1)
    pxx, pxy = _centroid(plus_x, 1)
    pyx, pyy = _centroid(plus_y, 1)
    assert pxx < cx
    assert abs(pxy - cy) < 0.25
    assert pyy > cy
    assert abs(pyx - cx) < 0.25


def test_mesh_renderer_is_deterministic():
    renderer = MeshOrthographicRenderer(CameraConfig(renderer_backend="mesh_orthographic"))
    pose = np.asarray([0.002, -0.003, 7.0], dtype=np.float32)
    a = renderer.render(pose, "square-concave2")
    b = renderer.render(pose, "square-concave2")
    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.mask, b.mask)
