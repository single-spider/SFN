from __future__ import annotations

from unittest.mock import patch

import numpy as np
from sfn.config import CameraConfig
from sfn.panda import PandaConfig
from sfn.panda.panda_scene import PandaScene


def test_mesh_derived_peg_really_extends_downward_toward_fixture():
    with PandaScene("square-concave1", PandaConfig(mesh_derived_alignment_z=True)) as scene:
        state = scene.measure()
        aabb = scene.p.getAABB(scene.ids.peg, physicsClientId=scene.client_id)
        assert state.peg_tip_pos_world[2] < state.peg_pos_world[2]
        assert aabb[0][2] < state.peg_pos_world[2] < aabb[1][2]
        assert abs(state.peg_tip_pos_world[2] - 0.0005) < 0.0002


def test_rigid_attachment_seats_peg_between_closed_fingers():
    with PandaScene("square-square", PandaConfig(mesh_derived_alignment_z=True)) as scene:
        p = scene.p
        cid = scene.client_id
        peg_min, peg_max = p.getAABB(scene.ids.peg, physicsClientId=cid)
        finger_aabbs = [
            p.getAABB(scene.ids.robot, joint, physicsClientId=cid)
            for joint in scene.config.finger_joint_indices
        ]

        # The peg top must occupy the fingers' vertical grasping region rather
        # than hanging below it with the former 66 mm air gap.
        finger_z_min = max(aabb[0][2] for aabb in finger_aabbs)
        finger_z_max = min(aabb[1][2] for aabb in finger_aabbs)
        assert finger_z_min < peg_max[2] < finger_z_max

        # At this tool yaw the fingers oppose one another along world X.  Their
        # inner gap is narrower than the peg body, giving a visible clamp while
        # the fixed constraint remains the authoritative attachment model.
        fingers_by_x = sorted(finger_aabbs, key=lambda aabb: aabb[0][0])
        inner_gap = fingers_by_x[1][0][0] - fingers_by_x[0][1][0]
        peg_width_x = peg_max[0] - peg_min[0]
        assert 0.0 < inner_gap < peg_width_x


def test_native_mask_labels_hole_mesh_not_fixture_body():
    config = PandaConfig(
        mesh_derived_alignment_z=True,
        camera_ignore_robot_occlusion=True,
        camera_eye_offset_m=(0.0, 0.0, 0.2),
        camera_target_offset_m=(0.0, 0.0, 0.03),
    )
    with PandaScene("square-square", config) as scene:
        scene.reset_to_pose_error([0.005, 0.0, 5.0])
        rendered = scene.render_camera(CameraConfig(far=0.5))
        assert int(np.sum(rendered.mask == 1)) > 0
        assert int(np.sum(rendered.mask == 2)) > 0
        # If the entire fixture were still labelled as seam this would occupy
        # thousands of pixels rather than only the exposed opening patch.
        assert int(np.sum(rendered.mask == 2)) < int(np.sum(rendered.mask == 1))
        assert rendered.metadata["semantic_source"] == "body_id_separate_hole_mesh"
        assert rendered.metadata["robot_occlusion_ignored"] is True
        assert rendered.metadata["physical_scene_unchanged"] is True


def test_robot_free_camera_does_not_change_visible_panda_state():
    config = PandaConfig(
        gui=False,
        mesh_derived_alignment_z=True,
        camera_ignore_robot_occlusion=True,
        camera_eye_offset_m=(0.0, 0.0, 0.2),
        camera_target_offset_m=(0.0, 0.0, 0.03),
    )
    with PandaScene("square-square", config) as scene:
        ids = scene.ids
        assert ids is not None
        before_state = scene.measure()
        before_visuals = [tuple(item[7]) for item in scene.p.getVisualShapeData(ids.robot, physicsClientId=scene.client_id)]
        rendered = scene.render_camera(CameraConfig(far=0.5))
        after_state = scene.measure()
        after_visuals = [tuple(item[7]) for item in scene.p.getVisualShapeData(ids.robot, physicsClientId=scene.client_id)]
        assert int(np.sum(rendered.mask == 1)) > 0
        assert np.allclose(after_state.joint_positions, before_state.joint_positions)
        assert np.allclose(after_state.peg_pos_world, before_state.peg_pos_world)
        assert after_visuals == before_visuals


def test_motion_observer_receives_multiple_measured_substeps():
    samples = []
    with PandaScene("square-square", PandaConfig(execution_mode="dynamic", mesh_derived_alignment_z=True)) as scene:
        scene.execute_cartesian_delta(
            0.0002,
            0.0,
            0.0,
            physics_observer=lambda state, index, total: samples.append((state, index, total)),
            observer_stride=30,
        )
    assert len(samples) == 4
    assert [index for _state, index, _total in samples] == [30, 60, 90, 120]
    assert all(total == 120 for _state, _index, total in samples)


def test_dynamic_command_does_not_reset_joints_or_teleport_peg():
    with PandaScene(
        "square-square",
        PandaConfig(execution_mode="dynamic", mesh_derived_alignment_z=True),
    ) as scene:
        with (
            patch.object(scene.p, "resetJointState", side_effect=AssertionError("dynamic joint reset")),
            patch.object(
                scene.p, "resetBasePositionAndOrientation", side_effect=AssertionError("dynamic peg teleport")
            ),
        ):
            result = scene.execute_cartesian_delta(0.0002, 0.0, 0.0)
        assert result.execution_mode == "dynamic"


def test_scene_close_disconnects_and_initial_state_has_no_unexpected_contacts():
    scene = PandaScene("square-square", PandaConfig(mesh_derived_alignment_z=True))
    client = scene.client_id
    assert scene.p.getConnectionInfo(client)["isConnected"]
    contacts = scene.contact_summary()
    assert not any(row.get("unexpected", False) for row in contacts)
    scene.close()
    assert not scene.p.getConnectionInfo(client)["isConnected"]
