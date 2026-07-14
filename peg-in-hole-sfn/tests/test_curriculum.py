import json

import pytest

pytest.importorskip("torch")

from sfn.config import CameraConfig, EnvironmentConfig
from sfn.training.curriculum import SFMSCurriculumStage, run_sfms_curriculum, stages_from_mapping


def test_curriculum_schema_is_strict():
    stages = stages_from_mapping({"stages": [{"name": "small", "updates": 1}]})
    assert stages[0].name == "small"
    with pytest.raises(ValueError, match="unknown"):
        stages_from_mapping({"stages": [{"name": "bad", "updates": 1, "surprise": True}]})
    with pytest.raises(ValueError, match="unique"):
        stages_from_mapping({"stages": [{"name": "x", "updates": 1}, {"name": "x", "updates": 1}]})


def test_tiny_curriculum_runs_and_hands_checkpoint_forward(tmp_path):
    report = run_sfms_curriculum(
        out_dir=tmp_path,
        stages=[
            SFMSCurriculumStage(
                name="small_xy",
                updates=1,
                rollout_steps=2,
                xy_initial_range_mm=1.0,
                yaw_initial_range_deg=0.0,
            )
        ],
        seed=70,
        shapes=["synthetic-square"],
        eval_shapes=["synthetic-square"],
        environment=EnvironmentConfig(),
        camera=CameraConfig(),
        segmentation_path=None,
        position_path=None,
        orientation_path=None,
        device="cpu",
    )
    assert (tmp_path / "01_small_xy" / "policy.pt").is_file()
    assert report["selected_checkpoint"].endswith("best.pt")
    persisted = json.loads((tmp_path / "curriculum_summary.json").read_text(encoding="utf-8"))
    assert persisted["stages"][0]["stage"]["name"] == "small_xy"
