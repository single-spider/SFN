"""Static Panda model metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class JointMetadata:
    index: int
    name: str
    joint_type: int
    lower: float
    upper: float
    link_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_joint_metadata(pybullet_module, robot_id: int, physics_client_id: int) -> list[JointMetadata]:
    rows: list[JointMetadata] = []
    for j in range(pybullet_module.getNumJoints(robot_id, physicsClientId=physics_client_id)):
        info = pybullet_module.getJointInfo(robot_id, j, physicsClientId=physics_client_id)
        rows.append(
            JointMetadata(
                j,
                info[1].decode("utf-8"),
                int(info[2]),
                float(info[8]),
                float(info[9]),
                info[12].decode("utf-8"),
            )
        )
    return rows
