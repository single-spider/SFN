"""Peg attachment data and drift metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PegAttachmentConfig:
    parent_link: int
    child_body: int
    parent_frame_pos: tuple[float, float, float]
    parent_frame_orn: tuple[float, float, float, float]
    child_frame_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    child_frame_orn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttachmentDrift:
    translation_mm: float
    yaw_deg: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "translation_mm": float(self.translation_mm),
            "yaw_deg": float(self.yaw_deg),
            "samples": int(self.samples),
        }
