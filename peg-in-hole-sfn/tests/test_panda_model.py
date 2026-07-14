from __future__ import annotations

import numpy as np
from sfn.panda import PandaScene


def test_panda_scene_loads_and_reports_measured_pose():
    with PandaScene("square-concave1") as scene:
        meta = scene.metadata()
        state = scene.measure()
        assert meta["robot_id"] >= 0
        assert meta["peg_id"] >= 0
        assert meta["base_id"] >= 0
        assert len(meta["joints"]) >= 12
        assert np.all(np.isfinite(state.pose_error_task))
        assert abs(state.pose_error_task[0]) < 2e-4
        assert abs(state.pose_error_task[1]) < 2e-4
        assert abs(state.pose_error_task[2]) < 1e-2
