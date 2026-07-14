"""PyBullet Panda + peg + hole scene with measured command execution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import CameraConfig
from ..constants import MASK_PEG, MASK_SEAM
from ..envs.renderer import RenderOutput
from .command import ExecutionResult
from .config import PandaConfig, TaskToWorldTransform, default_asset_root
from .kinematics import joint_limit_arrays
from .measurement import MeasuredPandaState, pose7, wrap_deg, yaw_from_quat_deg
from .peg_attachment import AttachmentDrift, PegAttachmentConfig
from .robot_model import JointMetadata, read_joint_metadata


class PandaDependencyError(RuntimeError):
    pass


def _require_pybullet():
    try:
        import pybullet as p
        import pybullet_data
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
        raise PandaDependencyError("pybullet and pybullet_data are required for Panda validation") from exc
    return p, pybullet_data


@dataclass
class PandaSceneIds:
    robot: int
    peg: int
    base: int
    seam: int
    constraint: int


class PandaScene:
    """Owns one PyBullet connection and exposes the SFN Panda action bridge."""

    def __init__(
        self,
        shape: str = "square-concave1",
        config: PandaConfig | None = None,
        task_transform: TaskToWorldTransform | None = None,
        asset_root: str | Path | None = None,
        seed: int = 0,
    ):
        self.shape = str(shape)
        self.config = config or PandaConfig()
        self.config.validate()
        self.task_transform = task_transform or TaskToWorldTransform()
        self.asset_root = Path(asset_root).resolve() if asset_root is not None else default_asset_root()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.p, self.pybullet_data = _require_pybullet()
        self.client_id = self.p.connect(self.p.GUI if self.config.gui else self.p.DIRECT)
        if self.client_id < 0:
            raise PandaDependencyError("Could not connect to PyBullet")
        self.ids: PandaSceneIds | None = None
        self.attachment: PegAttachmentConfig | None = None
        self.joint_metadata: list[JointMetadata] = []
        self._last_commanded_pose_error = np.zeros(3, dtype=np.float64)
        self._last_ik_diagnostics: dict[str, float | str | None] = {"residual_m": None, "branch": None}
        self.alignment_z_m = float(self.config.alignment_z_m)
        self.peg_tip_offset_m = float(self.config.peg_tip_offset_m)
        self.base_top_z_m = 0.0
        # The controller's deliberately robot-free top-down observation is
        # rendered by a separate DIRECT client.  Keeping it out of the GUI
        # client prevents the visible Panda from flickering while an image is
        # captured.
        self._perception_scene: PandaScene | None = None
        self._load_scene()
        if self.config.camera_ignore_robot_occlusion:
            perception_config = replace(
                self.config,
                gui=False,
                camera_ignore_robot_occlusion=False,
            )
            self._perception_scene = PandaScene(
                shape=self.shape,
                config=perception_config,
                task_transform=self.task_transform,
                asset_root=self.asset_root,
                seed=self.seed,
            )
            self._perception_scene._hide_robot_from_camera_once()

    @property
    def shape_dir(self) -> Path:
        return (self.asset_root / self.shape).resolve()

    @property
    def base_urdf(self) -> Path:
        return self.shape_dir / "base" / "base.urdf"

    @property
    def peg_urdf(self) -> Path:
        return self.shape_dir / "peg" / "peg_test.urdf"

    def _decomposed_peg_urdf(self) -> Path:
        """Return a cached compound-convex collision URDF for this peg."""
        source = self.shape_dir / "peg" / "peg.obj"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        cache = (
            Path(__file__).resolve().parents[2] / ".cache" / "panda_collision" / f"{self.shape}-{digest}-margincomp-v5"
        )
        urdf = cache / "peg_compound.urdf"
        if urdf.exists():
            return urdf
        import trimesh

        cache.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load_mesh(str(source), force="mesh", process=True)
        pieces = mesh.convex_decomposition(maxConvexHulls=32, resolution=100_000)
        if not isinstance(pieces, (list, tuple)):
            pieces = [pieces]
        collisions = []
        # Bullet gives every dynamic convex mesh an approximately 2 mm hull
        # margin which cannot be removed through changeDynamics. Compensate the
        # exported collision geometry so the effective collision envelope
        # matches the supplied OBJ dimensions (visual geometry is untouched).
        extent = np.asarray(mesh.extents, dtype=np.float64)
        margin_comp = 0.004
        xy_scale = np.maximum((extent[:2] - 2.0 * margin_comp) / np.maximum(extent[:2], 1e-9), 0.25)
        for index, piece in enumerate(pieces):
            piece = piece.copy()
            piece.vertices[:, 0] *= float(xy_scale[0])
            piece.vertices[:, 1] *= float(xy_scale[1])
            piece.vertices[:, 2] += 0.003
            part = cache / f"part_{index:03d}.obj"
            piece.export(part)
            collisions.append(f'<collision><geometry><mesh filename="{part.as_posix()}"/></geometry></collision>')
        urdf.write_text(
            '<?xml version="1.0"?><robot name="compound_peg"><link name="peg">'
            '<inertial><mass value="0.1"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>'
            f'<visual><geometry><mesh filename="{source.resolve().as_posix()}"/></geometry><material name="peg"><color rgba="0.91 0.91 0.91 1"/></material></visual>'
            + "".join(collisions)
            + "</link></robot>",
            encoding="utf-8",
        )
        return urdf

    def _create_raster_compound_peg(self, pose: np.ndarray) -> int:
        """Create a dynamic extruded-silhouette peg without convex-mesh margins.

        Bullet expands dynamic convex mesh hulls by a large implicit margin,
        which is unacceptable for sub-millimetre insertion.  The supplied peg
        is an extrusion, so a union of thin native box primitives reproduces
        its XY silhouette while preserving exact, margin-stable contact.
        """
        import cv2
        import trimesh

        source = self.shape_dir / "peg" / "peg.obj"
        mesh = trimesh.load_mesh(str(source), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        resolution = 0.00020  # 0.2 mm; boundary error is at most 0.1 mm.
        low = vertices[:, :2].min(axis=0) - resolution
        high = vertices[:, :2].max(axis=0) + resolution
        width, height = np.ceil((high - low) / resolution).astype(int) + 1
        mask = np.zeros((int(height), int(width)), dtype=np.uint8)
        pixel_xy = (vertices[:, :2] - low) / resolution
        for face in faces:
            cv2.fillConvexPoly(mask, np.rint(pixel_xy[face]).astype(np.int32), 1)

        # Merge vertically adjacent rows with the same run to keep the compound
        # small while retaining the full non-convex silhouette.
        active: dict[tuple[int, int], tuple[int, int]] = {}
        rectangles: list[tuple[int, int, int, int]] = []
        for row in range(mask.shape[0] + 1):
            runs: set[tuple[int, int]] = set()
            if row < mask.shape[0]:
                xs = np.flatnonzero(mask[row])
                if len(xs):
                    splits = np.flatnonzero(np.diff(xs) > 1)
                    starts = np.r_[0, splits + 1]
                    ends = np.r_[splits, len(xs) - 1]
                    runs = {(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends, strict=False)}
            for run in list(active):
                if run not in runs:
                    first, last = active.pop(run)
                    rectangles.append((run[0], run[1], first, last))
            for run in runs:
                if run in active:
                    active[run] = (active[run][0], row)
                else:
                    active[run] = (row, row)

        z_low, z_high = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        shape_types, half_extents, positions, orientations = [], [], [], []
        for x0, x1, y0, y1 in rectangles:
            x_min, x_max = low[0] + (x0 - 0.5) * resolution, low[0] + (x1 + 0.5) * resolution
            y_min, y_max = low[1] + (y0 - 0.5) * resolution, low[1] + (y1 + 0.5) * resolution
            shape_types.append(self.p.GEOM_BOX)
            half_extents.append([(x_max - x_min) / 2, (y_max - y_min) / 2, (z_high - z_low) / 2])
            positions.append([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_high + z_low) / 2])
            orientations.append([0, 0, 0, 1])
        collision = self.p.createCollisionShapeArray(
            shapeTypes=shape_types,
            halfExtents=half_extents,
            collisionFramePositions=positions,
            collisionFrameOrientations=orientations,
            physicsClientId=self.client_id,
        )
        visual = self.p.createVisualShape(
            shapeType=self.p.GEOM_MESH,
            fileName=str(source),
            rgbaColor=list(self.config.peg_rgba),
            physicsClientId=self.client_id,
        )
        return self.p.createMultiBody(
            baseMass=float(self.config.peg_mass_kg),
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=pose[:3],
            baseOrientation=pose[3:],
            physicsClientId=self.client_id,
        )

    def _load_scene(self) -> None:
        p = self.p
        cid = self.client_id
        p.setAdditionalSearchPath(self.pybullet_data.getDataPath(), physicsClientId=cid)
        p.setAdditionalSearchPath(str(self.shape_dir), physicsClientId=cid)
        p.setTimeStep(self.config.time_step, physicsClientId=cid)
        p.setGravity(*self.config.gravity, physicsClientId=cid)
        if not self.base_urdf.exists():
            raise FileNotFoundError(f"Missing base URDF for shape {self.shape}: {self.base_urdf}")
        if not self.peg_urdf.exists():
            raise FileNotFoundError(f"Missing standalone peg URDF for shape {self.shape}: {self.peg_urdf}")
        self._derive_mesh_reference_geometry()

        robot_urdf = self.config.robot_urdf
        if not Path(robot_urdf).is_absolute():
            candidate = Path(self.pybullet_data.getDataPath()) / robot_urdf
            robot_urdf = str(candidate if candidate.exists() else robot_urdf)
        robot = p.loadURDF(
            robot_urdf,
            basePosition=self.config.robot_base_pos,
            baseOrientation=p.getQuaternionFromEuler([math.radians(v) for v in self.config.robot_base_orn_euler_deg]),
            useFixedBase=True,
            physicsClientId=cid,
        )
        base_mesh_path = self.shape_dir / "base" / "base.obj"
        base_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH, fileName=str(base_mesh_path), rgbaColor=[0.75, 0.75, 0.75, 1.0], physicsClientId=cid
        )
        # The URDF's non-standard `concave="yes"` attribute is ignored by
        # PyBullet and can collapse the fixture into a solid convex hull. A
        # fixed concave trimesh preserves the actual through-opening.
        base_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=str(base_mesh_path),
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            physicsClientId=cid,
        )
        base = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=self.task_transform.origin_world,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=cid,
        )
        p.changeDynamics(base, -1, collisionMargin=float(self.config.collision_margin_m), physicsClientId=cid)
        p.changeDynamics(base, -1, lateralFriction=float(self.config.fixture_lateral_friction), physicsClientId=cid)
        # A collision-free copy of mask.obj sits in the physical opening.  It
        # gives the native camera a distinct *hole/seam* object ID instead of
        # incorrectly labelling the complete fixture body as seam.
        import trimesh

        seam_mesh = trimesh.load_mesh(str(self.shape_dir / "mask.obj"), force="mesh", process=False)
        seam_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=np.asarray(seam_mesh.vertices, dtype=np.float64).tolist(),
            indices=np.asarray(seam_mesh.faces, dtype=np.int32).reshape(-1).tolist(),
            meshScale=[1.0, 1.0, 1.0],
            rgbaColor=[0.08, 0.08, 0.08, 1.0],
            physicsClientId=cid,
        )
        seam = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=seam_visual,
            baseCollisionShapeIndex=-1,
            # Raise the semantic insert by 1 mm to avoid coplanar z-fighting
            # with the fixture top in TinyRenderer. It remains collision-free.
            basePosition=(
                np.asarray(self.task_transform.origin_world, dtype=np.float64) + np.asarray([0.0, 0.0, 1e-3])
            ).tolist(),
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=cid,
        )
        self.ids = PandaSceneIds(robot=robot, peg=-1, base=base, seam=seam, constraint=-1)
        self.joint_metadata = read_joint_metadata(p, robot, cid)
        self._reset_arm_to_pose_error(np.zeros(3, dtype=np.float64))

        peg_pose = self._expected_peg_pose_from_ee()
        if self.config.use_convex_decomposition:
            peg = self._create_raster_compound_peg(peg_pose)
        else:
            peg = p.loadURDF(
                str(self.peg_urdf),
                basePosition=peg_pose[:3],
                baseOrientation=peg_pose[3:],
                useFixedBase=False,
                physicsClientId=cid,
            )
        p.changeDynamics(peg, -1, collisionMargin=float(self.config.collision_margin_m), physicsClientId=cid)
        p.changeVisualShape(peg, -1, rgbaColor=list(self.config.peg_rgba), physicsClientId=cid)
        p.changeDynamics(peg, -1, lateralFriction=float(self.config.peg_lateral_friction), physicsClientId=cid)
        for link in range(-1, p.getNumJoints(robot, physicsClientId=cid)):
            p.setCollisionFilterPair(robot, peg, link, -1, 0, physicsClientId=cid)

        parent_orn = p.getQuaternionFromEuler([0.0, 0.0, math.radians(self.config.attach_parent_frame_yaw_deg)])
        child_orn = p.getQuaternionFromEuler([math.radians(v) for v in self.config.attach_child_frame_rpy_deg])
        constraint = p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=self.config.ee_link_index,
            childBodyUniqueId=peg,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=self.config.attach_parent_frame_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=parent_orn,
            childFrameOrientation=child_orn,
            physicsClientId=cid,
        )
        self.ids = PandaSceneIds(robot=robot, peg=peg, base=base, seam=seam, constraint=constraint)
        self.attachment = PegAttachmentConfig(
            parent_link=self.config.ee_link_index,
            child_body=peg,
            parent_frame_pos=self.config.attach_parent_frame_pos,
            parent_frame_orn=tuple(parent_orn),
        )
        self._sync_peg_to_expected()

    def close(self) -> None:
        if self._perception_scene is not None:
            self._perception_scene.close()
            self._perception_scene = None
        if getattr(self, "client_id", None) is not None and self.client_id >= 0:
            try:
                self.p.disconnect(self.client_id)
            finally:
                self.client_id = -1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _target_quat(self, yaw_deg: float):
        r, pch, y = self.config.tool_roll_pitch_yaw_deg
        return self.p.getQuaternionFromEuler([math.radians(r), math.radians(pch), math.radians(y + float(yaw_deg))])

    def _target_pos(self, pose_error_task: Iterable[float], z_m: float | None = None) -> np.ndarray:
        pose = np.asarray(pose_error_task, dtype=np.float64).reshape(3)
        return self.task_transform.task_pose_to_world_pos(
            float(pose[0]),
            float(pose[1]),
            self.alignment_z_m if z_m is None else float(z_m),
        )

    def _derive_mesh_reference_geometry(self) -> None:
        """Derive peg-tip and approach Z from the actual mesh reference frame."""
        try:
            import trimesh
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise PandaDependencyError("trimesh is required for Panda mesh reference validation") from exc
        peg = trimesh.load_mesh(str(self.shape_dir / "peg" / "peg.obj"), force="mesh", process=False)
        base = trimesh.load_mesh(str(self.shape_dir / "base" / "base.obj"), force="mesh", process=False)
        peg_vertices = np.asarray(peg.vertices, dtype=np.float64)
        base_vertices = np.asarray(base.vertices, dtype=np.float64)
        peg_min_z = float(np.min(peg_vertices[:, 2]))
        self.peg_tip_offset_m = float(-peg_min_z)
        self.base_top_z_m = float(np.max(base_vertices[:, 2]))
        if self.config.mesh_derived_alignment_z:
            self.alignment_z_m = self.base_top_z_m - peg_min_z + float(self.config.alignment_clearance_mm) / 1000.0

    def _target_ee_pos(self, pose_error_task: Iterable[float]) -> np.ndarray:
        pose = np.asarray(pose_error_task, dtype=np.float64).reshape(3)
        desired_peg_pos = self._target_pos(pose)

        # The fixed attachment gives
        #   peg_position = ee_position + R(ee) * parent_frame_position
        # because the child-frame translation is zero.  Invert that relation
        # so changing the visual grasp transform never changes the commanded
        # peg reference or insertion target.
        ee_quat = self._target_quat(float(pose[2]))
        rotation = np.asarray(self.p.getMatrixFromQuaternion(ee_quat), dtype=np.float64).reshape(3, 3)
        parent_offset_world = rotation @ np.asarray(
            self.config.attach_parent_frame_pos, dtype=np.float64
        )
        return desired_peg_pos - parent_offset_world

    def solve_ik(
        self,
        target_pos_world: np.ndarray,
        target_quat_world: np.ndarray,
        current_joint_positions: np.ndarray | None = None,
        allow_stateful_search: bool = False,
    ) -> np.ndarray:
        ids = self._require_ids(allow_prepeg=True)
        p = self.p
        cid = self.client_id
        lowers, uppers, ranges = joint_limit_arrays(p, ids.robot, self.config.arm_joint_indices, cid)
        rest = (
            list(np.asarray(current_joint_positions, dtype=np.float64).reshape(-1)[:7])
            if current_joint_positions is not None
            else list(self.config.rest_poses)
        )
        target_pos = np.asarray(target_pos_world, dtype=np.float64).reshape(3)
        target_quat = np.asarray(target_quat_world, dtype=np.float64).reshape(4)
        if (
            self.config.execution_mode == "dynamic"
            and current_joint_positions is not None
            and not allow_stateful_search
        ):
            # Dynamic execution must never alter the measured robot merely to
            # search IK branches. PyBullet uses the current state internally;
            # a null-space solve with the measured joints as its rest pose is
            # therefore the honest non-mutating command path.
            raw = p.calculateInverseKinematics(
                ids.robot,
                self.config.ee_link_index,
                target_pos.tolist(),
                target_quat.tolist(),
                lowerLimits=lowers,
                upperLimits=uppers,
                jointRanges=ranges,
                restPoses=rest,
                maxNumIterations=self.config.ik_max_iterations,
                residualThreshold=self.config.ik_residual_threshold,
                physicsClientId=cid,
            )
            solution = np.asarray(raw[: len(self.config.arm_joint_indices)], dtype=np.float64)
            if np.any(solution < np.asarray(lowers) - 1e-7) or np.any(solution > np.asarray(uppers) + 1e-7):
                raise RuntimeError("Panda IK could not find a joint-limit-feasible dynamic solution")
            self._last_ik_diagnostics = {"residual_m": None, "branch": "measured_rest"}
            return solution
        saved = [p.getJointState(ids.robot, int(j), physicsClientId=cid)[0] for j in self.config.arm_joint_indices]
        rest_candidates = [
            rest,
            list(self.config.rest_poses),
            [rest[0], rest[1], rest[2], -2.3, rest[4], 2.7, rest[6]],
            [-0.35, 1.8, -1.1, 0.8, -2.1, -1.4, -0.4],
        ]
        best_raw = None
        best_score = None
        best_branch = None
        for branch_index, rest_pose in enumerate(rest_candidates):
            # PyBullet IK uses current joint state as part of its iterative
            # initialization. Seed each branch from its own null-space rest
            # pose; restoring the original state here makes restPoses largely
            # ineffective and can select an unreachable/limit-violating branch.
            for joint_index, q in zip(self.config.arm_joint_indices, rest_pose, strict=False):
                p.resetJointState(ids.robot, int(joint_index), float(q), physicsClientId=cid)
            raw = p.calculateInverseKinematics(
                ids.robot,
                self.config.ee_link_index,
                target_pos.tolist(),
                target_quat.tolist(),
                lowerLimits=lowers,
                upperLimits=uppers,
                jointRanges=ranges,
                restPoses=rest_pose,
                maxNumIterations=self.config.ik_max_iterations,
                residualThreshold=self.config.ik_residual_threshold,
                physicsClientId=cid,
            )
            for joint_index, q in zip(
                self.config.arm_joint_indices, raw[: len(self.config.arm_joint_indices)], strict=False
            ):
                p.resetJointState(ids.robot, int(joint_index), float(q), physicsClientId=cid)
            actual = np.asarray(
                p.getLinkState(
                    ids.robot, self.config.ee_link_index, computeForwardKinematics=True, physicsClientId=cid
                )[0],
                dtype=np.float64,
            )
            raw_arm = np.asarray(raw[: len(self.config.arm_joint_indices)], dtype=np.float64)
            below = np.maximum(np.asarray(lowers, dtype=np.float64) - raw_arm, 0.0)
            above = np.maximum(raw_arm - np.asarray(uppers, dtype=np.float64), 0.0)
            violation = below + above
            pos_err = float(np.linalg.norm(actual - target_pos))
            limit_bad = bool(np.any(violation > 1e-7))
            # Feasible solutions always outrank infeasible ones; position error
            # then selects among feasible IK branches.
            score = (
                int(limit_bad),
                float(violation.sum()),
                pos_err,
                float(np.linalg.norm(raw_arm - np.asarray(rest_pose))),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_raw = raw
                best_branch = f"rest_candidate_{branch_index}"
        for joint_index, q in zip(self.config.arm_joint_indices, saved, strict=False):
            p.resetJointState(ids.robot, int(joint_index), float(q), physicsClientId=cid)
        if best_raw is None:
            raise RuntimeError("Panda IK returned no candidate")
        solution = np.asarray(best_raw[: len(self.config.arm_joint_indices)], dtype=np.float64)
        if np.any(solution < np.asarray(lowers) - 1e-7) or np.any(solution > np.asarray(uppers) + 1e-7):
            raise RuntimeError("Panda IK could not find a joint-limit-feasible solution")
        self._last_ik_diagnostics = {
            "residual_m": None if best_score is None else float(best_score[2]),
            "branch": best_branch,
        }
        return solution

    def execute_joint_target(
        self,
        joint_target: np.ndarray,
        commanded_pose_error: np.ndarray | None = None,
        steps: int | None = None,
        physics_observer: Callable[[MeasuredPandaState, int, int], None] | None = None,
        observer_stride: int = 8,
    ) -> ExecutionResult:
        ids = self._require_ids()
        p = self.p
        cid = self.client_id
        joint_target = np.asarray(joint_target, dtype=np.float64).reshape(len(self.config.arm_joint_indices))
        commanded_pose_error = (
            self._last_commanded_pose_error
            if commanded_pose_error is None
            else np.asarray(commanded_pose_error, dtype=np.float64).reshape(3)
        )
        current_joint = np.asarray(
            [p.getJointState(ids.robot, int(j), physicsClientId=cid)[0] for j in self.config.arm_joint_indices],
            dtype=np.float64,
        )
        for joint_index in self.config.finger_joint_indices:
            p.setJointMotorControl2(
                ids.robot,
                int(joint_index),
                p.POSITION_CONTROL,
                targetPosition=self.config.finger_open,
                force=20.0,
                physicsClientId=cid,
            )
        total_steps = int(steps or self.config.command_steps)
        if observer_stride < 1:
            raise ValueError("observer_stride must be at least one")
        for step_index in range(total_steps):
            # Dynamic commands are ramped from the measured state, avoiding an
            # instantaneous joint-target discontinuity. Kinematic mode uses
            # the same ramp before its explicitly labelled idealized reset.
            alpha = float(step_index + 1) / float(max(total_steps, 1))
            waypoint = current_joint + alpha * (joint_target - current_joint)
            for joint_index, q in zip(self.config.arm_joint_indices, waypoint, strict=False):
                p.setJointMotorControl2(
                    ids.robot,
                    int(joint_index),
                    p.POSITION_CONTROL,
                    targetPosition=float(q),
                    force=float(self.config.max_force),
                    positionGain=float(self.config.position_gain),
                    velocityGain=float(self.config.velocity_gain),
                    physicsClientId=cid,
                )
            p.stepSimulation(physicsClientId=cid)
            if physics_observer is not None and (
                (step_index + 1) % int(observer_stride) == 0 or step_index + 1 == total_steps
            ):
                physics_observer(self.measure(), step_index + 1, total_steps)
        if self.config.execution_mode == "kinematic":
            # Explicit idealized mode: remove motor settling residue for pure
            # coordinate/IK validation. Never report this as dynamic tracking.
            for joint_index, q in zip(self.config.arm_joint_indices, joint_target, strict=False):
                p.resetJointState(ids.robot, int(joint_index), float(q), physicsClientId=cid)
        self._last_commanded_pose_error = commanded_pose_error.astype(np.float64)
        if self.config.execution_mode == "kinematic":
            self._sync_peg_to_expected()

        measured = self.measure()
        actual = measured.joint_positions
        joint_limit_margins = self.joint_limit_margins(actual)
        commanded_peg_pose = pose7(
            self._target_pos(commanded_pose_error), self._target_quat(float(commanded_pose_error[2]))
        )
        measured_peg_pose = pose7(measured.peg_pos_world, measured.peg_quat_world)
        pos_error_m = float(np.linalg.norm(measured.peg_pos_world[:2] - commanded_peg_pose[:2]))
        yaw_error_deg = abs(wrap_deg(float(measured.pose_error_task[2]) - float(commanded_pose_error[2])))
        return ExecutionResult(
            commanded_ee_pose=pose7(
                self._target_ee_pos(commanded_pose_error), self._target_quat(float(commanded_pose_error[2]))
            ),
            measured_ee_pose=pose7(measured.ee_pos_world, measured.ee_quat_world),
            commanded_peg_pose=commanded_peg_pose,
            measured_peg_pose=measured_peg_pose,
            joint_target=joint_target.copy(),
            joint_actual=actual.copy(),
            pos_error_m=pos_error_m,
            yaw_error_deg=float(yaw_error_deg),
            max_joint_error=float(np.max(np.abs(actual - joint_target))) if actual.size else 0.0,
            contacts=self.contact_summary(),
            ik_success=bool(pos_error_m <= 0.001 and yaw_error_deg <= 1.0),
            execution_mode=self.config.execution_mode,
            joint_limit_violation=bool(self._joint_limit_violation(actual)),
            joint_limit_margins=joint_limit_margins,
            ik_residual_m=self._last_ik_diagnostics.get("residual_m"),
            ik_branch=str(self._last_ik_diagnostics["branch"]) if self._last_ik_diagnostics.get("branch") else None,
        )

    def execute_cartesian_delta(
        self,
        dx_m: float,
        dy_m: float,
        dyaw_deg: float,
        physics_observer: Callable[[MeasuredPandaState, int, int], None] | None = None,
        observer_stride: int = 8,
    ) -> ExecutionResult:
        measured = self.measure()
        target_pose_error = np.asarray(measured.pose_error_task, dtype=np.float64) + np.asarray(
            [dx_m, dy_m, dyaw_deg], dtype=np.float64
        )
        joints = self.solve_ik(
            self._target_ee_pos(target_pose_error),
            self._target_quat(float(target_pose_error[2])),
            measured.joint_positions,
        )
        return self.execute_joint_target(
            joints,
            target_pose_error,
            physics_observer=physics_observer,
            observer_stride=observer_stride,
        )

    def reset_to_pose_error(self, pose_error_task: Iterable[float]) -> MeasuredPandaState:
        pose = np.asarray(pose_error_task, dtype=np.float64).reshape(3)
        self._reset_arm_to_pose_error(pose)
        self._last_commanded_pose_error = pose.copy()
        self._sync_peg_to_expected()
        return self.measure()

    def _reset_arm_to_pose_error(self, pose_error_task: np.ndarray) -> None:
        ids = self._require_ids(allow_prepeg=True)
        p = self.p
        cid = self.client_id
        joints = self.solve_ik(
            self._target_ee_pos(pose_error_task),
            self._target_quat(float(pose_error_task[2])),
            np.asarray(self.config.rest_poses, dtype=np.float64),
            allow_stateful_search=True,
        )
        for joint_index, q in zip(self.config.arm_joint_indices, joints, strict=False):
            p.resetJointState(ids.robot, int(joint_index), float(q), physicsClientId=cid)
        for joint_index in self.config.finger_joint_indices:
            p.resetJointState(ids.robot, int(joint_index), self.config.finger_open, physicsClientId=cid)
        self._last_commanded_pose_error = pose_error_task.astype(np.float64)

    def _expected_peg_pose_from_ee(self) -> np.ndarray:
        ee = self._ee_pose()
        parent_pos, parent_orn = self.p.multiplyTransforms(
            ee[:3].tolist(),
            ee[3:].tolist(),
            list(self.config.attach_parent_frame_pos),
            self.p.getQuaternionFromEuler([0, 0, math.radians(self.config.attach_parent_frame_yaw_deg)]),
            physicsClientId=self.client_id,
        )
        child_orn = self.p.getQuaternionFromEuler([math.radians(v) for v in self.config.attach_child_frame_rpy_deg])
        inv_child_pos, inv_child_orn = self.p.invertTransform(
            [0.0, 0.0, 0.0], child_orn, physicsClientId=self.client_id
        )
        pos, orn = self.p.multiplyTransforms(
            parent_pos, parent_orn, inv_child_pos, inv_child_orn, physicsClientId=self.client_id
        )
        return pose7(pos, orn)

    def _sync_peg_to_expected(self) -> None:
        ids = self._require_ids()
        peg_pose = self._expected_peg_pose_from_ee()
        self.p.resetBasePositionAndOrientation(
            ids.peg, peg_pose[:3].tolist(), peg_pose[3:].tolist(), physicsClientId=self.client_id
        )

    def _ee_pose(self) -> np.ndarray:
        ids = self._require_ids(allow_prepeg=True)
        st = self.p.getLinkState(
            ids.robot, self.config.ee_link_index, computeForwardKinematics=True, physicsClientId=self.client_id
        )
        return pose7(st[0], st[1])

    def measure(self) -> MeasuredPandaState:
        ids = self._require_ids()
        p = self.p
        cid = self.client_id
        joint_states = [p.getJointState(ids.robot, int(j), physicsClientId=cid) for j in self.config.arm_joint_indices]
        ee = self._ee_pose()
        peg_pos, peg_quat = p.getBasePositionAndOrientation(ids.peg, physicsClientId=cid)
        base_pos, base_quat = p.getBasePositionAndOrientation(ids.base, physicsClientId=cid)
        peg_pos = np.asarray(peg_pos, dtype=np.float64)
        peg_quat = np.asarray(peg_quat, dtype=np.float64)
        hole_pos = np.asarray(base_pos, dtype=np.float64)
        planar = self.task_transform.world_delta_to_task_delta(peg_pos - hole_pos)
        # Measure yaw from the grasp-target orientation.  The attached peg
        # quaternion is close to a roll/pitch singularity, so Euler-z on the
        # standalone peg body flips by ~180 deg even when the tool yaw is
        # correct.  The grasp-target link is the measured robot frame that the
        # peg is rigidly attached to.
        yaw0 = self.config.tool_roll_pitch_yaw_deg[2]
        yaw = wrap_deg(yaw_from_quat_deg(ee[3:], p) - yaw0)
        # The tool remains nominally vertical in this task. The offset is
        # derived from the actual OBJ bounds rather than the old 35 mm guess.
        peg_tip = peg_pos + np.asarray([0.0, 0.0, -self.peg_tip_offset_m], dtype=np.float64)
        return MeasuredPandaState(
            joint_positions=np.asarray([s[0] for s in joint_states], dtype=np.float64),
            joint_velocities=np.asarray([s[1] for s in joint_states], dtype=np.float64),
            ee_pos_world=ee[:3].copy(),
            ee_quat_world=ee[3:].copy(),
            peg_pos_world=peg_pos,
            peg_quat_world=peg_quat,
            peg_tip_pos_world=peg_tip,
            peg_tip_quat_world=peg_quat.copy(),
            hole_pos_world=hole_pos,
            hole_quat_world=np.asarray(base_quat, dtype=np.float64),
            pose_error_task=np.asarray([planar[0], planar[1], yaw], dtype=np.float64),
        )

    def contact_summary(self) -> list[dict]:
        ids = self._require_ids()
        contacts = []
        for c in self.p.getContactPoints(physicsClientId=self.client_id):
            if ids.peg in (c[1], c[2]) or ids.base in (c[1], c[2]):
                contacts.append(
                    {
                        "body_a": int(c[1]),
                        "body_b": int(c[2]),
                        "link_a": int(c[3]),
                        "link_b": int(c[4]),
                        "distance": float(c[8]),
                        "normal_force": float(c[9]),
                    }
                )
        return contacts

    def _joint_limit_violation(self, joints: np.ndarray) -> bool:
        ids = self._require_ids(allow_prepeg=True)
        lowers, uppers, _ = joint_limit_arrays(self.p, ids.robot, self.config.arm_joint_indices, self.client_id)
        q = np.asarray(joints, dtype=np.float64)
        return bool(np.any(q < np.asarray(lowers) - 1e-7) or np.any(q > np.asarray(uppers) + 1e-7))

    def joint_limit_margins(self, joints: np.ndarray) -> np.ndarray:
        """Signed distance in radians from each arm joint to its nearest limit."""
        ids = self._require_ids(allow_prepeg=True)
        lowers, uppers, _ = joint_limit_arrays(
            self.p, ids.robot, self.config.arm_joint_indices, self.client_id
        )
        q = np.asarray(joints, dtype=np.float64)
        return np.minimum(q - np.asarray(lowers), np.asarray(uppers) - q)

    def render_camera(self, camera_config: CameraConfig | None = None) -> RenderOutput:
        """Render native PyBullet RGB and body-ID segmentation mask.

        The mask uses the project convention: background=0, peg=1, visible
        hole/seam region=2.  It is intentionally body-ID based, not learned
        segmentation, so it can validate Panda camera geometry before sim-to-real
        segmentation training.
        """
        ids = self._require_ids()
        cam = camera_config or CameraConfig()
        if self.config.camera_ignore_robot_occlusion:
            return self._render_robot_free_camera(cam)
        p = self.p
        cid = self.client_id
        origin = np.asarray(self.task_transform.origin_world, dtype=np.float64)
        eye = origin + np.asarray(self.config.camera_eye_offset_m, dtype=np.float64)
        target = origin + np.asarray(self.config.camera_target_offset_m, dtype=np.float64)
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), list(self.config.camera_up_vector))
        proj = p.computeProjectionMatrixFOV(
            fov=float(cam.fov_y_deg),
            aspect=float(cam.crop_width) / float(cam.crop_height),
            nearVal=float(cam.near),
            farVal=float(cam.far),
        )
        _w, _h, rgba, _depth, seg = p.getCameraImage(
            width=int(cam.crop_width),
            height=int(cam.crop_height),
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=cid,
        )
        rgba_arr = np.asarray(rgba, dtype=np.uint8).reshape(int(cam.crop_height), int(cam.crop_width), 4)
        rgb = np.transpose(rgba_arr[:, :, :3], (2, 0, 1)).astype(np.uint8)
        seg_arr = np.asarray(seg, dtype=np.int64).reshape(int(cam.crop_height), int(cam.crop_width))
        object_ids = seg_arr & ((1 << 24) - 1)
        mask = np.zeros((int(cam.crop_height), int(cam.crop_width)), dtype=np.uint8)
        mask[object_ids == int(ids.seam)] = MASK_SEAM
        mask[object_ids == int(ids.peg)] = MASK_PEG
        return RenderOutput(
            rgb=rgb,
            mask=mask,
            metadata={
                "renderer_backend": "panda_native_pybullet",
                "semantic_source": "body_id_separate_hole_mesh",
                "asset_faithful": True,
                "robot_occlusion_ignored": False,
            },
        )

    def _hide_robot_from_camera_once(self) -> None:
        """Make a DIRECT-client robot invisible once, never in the GUI client."""
        ids = self._require_ids()
        for visual in self.p.getVisualShapeData(ids.robot, physicsClientId=self.client_id):
            rgba = tuple(float(value) for value in visual[7])
            self.p.changeVisualShape(
                ids.robot,
                int(visual[1]),
                rgbaColor=[*rgba[:3], 0.0],
                physicsClientId=self.client_id,
            )

    def _render_robot_free_camera(self, camera_config: CameraConfig) -> RenderOutput:
        """Render the calibrated controller view without touching GUI visuals."""
        if self._perception_scene is None:
            raise RuntimeError("robot-free perception scene was not initialized")
        source = self.measure()
        target = self._perception_scene
        target_ids = target._require_ids()
        target.p.resetBasePositionAndOrientation(
            target_ids.peg,
            source.peg_pos_world.tolist(),
            source.peg_quat_world.tolist(),
            physicsClientId=target.client_id,
        )
        rendered = target.render_camera(camera_config)
        rendered.metadata.update(
            {
                "renderer_backend": "panda_native_offscreen_observation",
                "robot_occlusion_ignored": True,
                "physical_scene_unchanged": True,
            }
        )
        return rendered

    def validate_attachment_drift(self, steps: int = 1000) -> AttachmentDrift:
        initial = self._relative_ee_peg()
        max_trans = 0.0
        max_yaw = 0.0
        for _ in range(int(steps)):
            current_q = self.measure().joint_positions
            for joint_index, q in zip(self.config.arm_joint_indices, current_q, strict=False):
                self.p.setJointMotorControl2(
                    self._require_ids().robot,
                    int(joint_index),
                    self.p.POSITION_CONTROL,
                    targetPosition=float(q),
                    force=float(self.config.max_force),
                    physicsClientId=self.client_id,
                )
            self.p.stepSimulation(physicsClientId=self.client_id)
            if self.config.execution_mode == "kinematic":
                for joint_index, q in zip(self.config.arm_joint_indices, current_q, strict=False):
                    self.p.resetJointState(
                        self._require_ids().robot, int(joint_index), float(q), physicsClientId=self.client_id
                    )
                self._sync_peg_to_expected()
            rel = self._relative_ee_peg()
            max_trans = max(max_trans, float(np.linalg.norm(rel[:3] - initial[:3]) * 1000.0))
            max_yaw = max(max_yaw, abs(wrap_deg(rel[3] - initial[3])))
        return AttachmentDrift(max_trans, max_yaw, int(steps))

    def _relative_ee_peg(self) -> np.ndarray:
        m = self.measure()
        delta = m.peg_pos_world - m.ee_pos_world
        yaw = wrap_deg(yaw_from_quat_deg(m.peg_quat_world, self.p) - yaw_from_quat_deg(m.ee_quat_world, self.p))
        return np.asarray([delta[0], delta[1], delta[2], yaw], dtype=np.float64)

    def _require_ids(self, allow_prepeg: bool = False) -> PandaSceneIds:
        if self.ids is None:
            raise RuntimeError("Panda scene has not loaded")
        if not allow_prepeg and self.ids.peg < 0:
            raise RuntimeError("Panda peg has not loaded")
        return self.ids

    def metadata(self) -> dict:
        ids = self._require_ids()
        return {
            "shape": self.shape,
            "client_id": self.client_id,
            "robot_id": ids.robot,
            "peg_id": ids.peg,
            "base_id": ids.base,
            "constraint_id": ids.constraint,
            "config": self.config.to_dict(),
            "execution_mode": self.config.execution_mode,
            "alignment_z_m": self.alignment_z_m,
            "peg_tip_offset_m": self.peg_tip_offset_m,
            "base_top_z_m": self.base_top_z_m,
            "joints": [j.to_dict() for j in self.joint_metadata],
            "attachment": None if self.attachment is None else self.attachment.to_dict(),
        }
