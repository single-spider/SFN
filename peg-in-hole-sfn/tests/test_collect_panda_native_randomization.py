import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scripts.collect_panda_native_dataset import _flush, _randomize_render

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "collect_panda_native_dataset.py"


def _render() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((3, 40, 50), dtype=np.uint8)
    rgb[:, 8:32, 12:38] = np.asarray([[[30]], [[100]], [[220]]], dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[8:20, 12:38] = 1
    mask[20:32, 12:38] = 2
    return rgb, mask


def test_panda_native_randomization_uses_shared_deterministic_contract():
    rgb, mask = _render()
    first = _randomize_render(rgb, mask, "heavy", dataset_seed=19, sample_id=7)
    second = _randomize_render(rgb, mask, "heavy", dataset_seed=19, sample_id=7)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["level"] == "heavy" and first[2]["seed"] is not None
    assert set(np.unique(first[1])) <= set(np.unique(mask)) == {0, 1, 2}


def test_panda_native_none_is_identity_but_still_has_replay_metadata():
    rgb, mask = _render()
    out_rgb, out_mask, record = _randomize_render(rgb, mask, "none", dataset_seed=3, sample_id=0)

    assert np.array_equal(out_rgb, rgb)
    assert np.array_equal(out_mask, mask)
    assert record["level"] == "none" and record["seed"] is not None


def test_panda_native_chunk_records_schema_v2_augmentation_json(tmp_path):
    rgb, mask = _render()
    rgb, mask, record = _randomize_render(rgb, mask, "medium", dataset_seed=11, sample_id=0)
    row = {
        "rgb": rgb,
        "mask": mask,
        "pose_error": np.zeros(3, dtype=np.float32),
        "position_target": np.zeros((21, 21), dtype=np.uint8),
        "orientation_index": 5,
        "shape_id": "square-square",
        "sample_id": 0,
        "seed": 11,
        "camera_variant": 0,
        "augmentation_json": json.dumps(record, sort_keys=True, separators=(",", ":")),
        "shape_family": "square",
        "symmetry_order": 4,
        "episode_id": 0,
        "frame_id": 0,
    }
    meta = _flush(tmp_path, "train_seen", 0, [row], compress=True)

    with np.load(tmp_path / meta["path"], allow_pickle=False) as arrays:
        stored = json.loads(str(arrays["augmentation_json"][0]))
        assert stored == record
        assert set(np.unique(arrays["mask"])) <= {0, 1, 2}


def test_panda_native_cli_exposes_all_randomization_levels():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--randomization-level {none,light,medium,heavy}" in result.stdout
