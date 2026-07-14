"""Robot-independent data contracts for guarded Cartesian commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


def _vector3(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values, got {result.shape}")
    return result


@dataclass(frozen=True)
class CartesianCommand:
    """A small tool-frame or world-frame Cartesian displacement."""

    translation_m: np.ndarray
    rotation_rad: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    frame: str = "tool"
    issued_at: float = 0.0
    sequence: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation_m", _vector3(self.translation_m, "translation_m"))
        object.__setattr__(self, "rotation_rad", _vector3(self.rotation_rad, "rotation_rad"))
        if not self.frame:
            raise ValueError("frame must not be empty")


@dataclass(frozen=True)
class RobotState:
    """Minimum feedback required by the software safety layer."""

    position_m: np.ndarray
    orientation_rad: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    measured_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vector3(self.position_m, "position_m"))
        object.__setattr__(self, "orientation_rad", _vector3(self.orientation_rad, "orientation_rad"))


@runtime_checkable
class CommandSink(Protocol):
    """Adapter boundary to a robot driver; no vendor types cross this boundary."""

    def send(self, command: CartesianCommand) -> None: ...

    def stop(self, reason: str) -> None: ...


class RecordingCommandSink:
    """Dry-run sink that records commands without touching hardware."""

    def __init__(self) -> None:
        self.commands: list[CartesianCommand] = []
        self.stop_reasons: list[str] = []

    def send(self, command: CartesianCommand) -> None:
        self.commands.append(command)

    def stop(self, reason: str) -> None:
        self.stop_reasons.append(reason)
