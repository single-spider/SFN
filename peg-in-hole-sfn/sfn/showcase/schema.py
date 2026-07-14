"""Versioned, safe telemetry contracts for the local Panda showcase service."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sfn.constants import ALL_EXPECTED_SHAPES

REPLAY_SCHEMA_ID = "sfn.showcase.replay/v2"
ALLOWED_SHAPES = frozenset(ALL_EXPECTED_SHAPES)
ALLOWED_METHODS = frozenset({"sfss", "sfms", "mfms"})


class SessionRequest(BaseModel):
    """A bounded autonomous Panda insertion request; no manual robot commands."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shape: str
    method: str = "sfms"
    seed: Annotated[int, Field(ge=0, le=2_147_483_647)] = 9900

    @field_validator("shape")
    @classmethod
    def allow_shape(cls, value: str) -> str:
        if value not in ALLOWED_SHAPES:
            raise ValueError(f"shape must be one of: {', '.join(sorted(ALLOWED_SHAPES))}")
        return value

    @field_validator("method")
    @classmethod
    def allow_method(cls, value: str) -> str:
        if value not in ALLOWED_METHODS:
            raise ValueError(f"method must be one of: {', '.join(sorted(ALLOWED_METHODS))}")
        return value


class SessionCommand(BaseModel):
    """Lifecycle-only commands accepted by an autonomous session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: Literal["start", "pause", "reset", "close"]


class TelemetrySample(BaseModel):
    """One measured state suitable for both live and replay 3-D viewers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: Annotated[int, Field(ge=0)]
    phase: Literal["ready", "alignment", "insertion", "complete", "failed"]
    joint_positions: tuple[float, ...]
    peg_pose: tuple[float, float, float, float, float, float, float]
    hole_pose: tuple[float, float, float, float, float, float, float]
    xy_error_mm: float
    yaw_error_deg: float
    action: tuple[float, float, float] | None = None
    insertion_depth_mm: float | None = None
    contact_count: Annotated[int, Field(ge=0)] = 0
    max_contact_force: Annotated[float, Field(ge=0.0)] = 0.0
    tracking_error_mm: Annotated[float, Field(ge=0.0)] = 0.0
    terminated: bool = False
    success: bool = False
    reason: str | None = None


class SessionEvent(BaseModel):
    """Server-originated lifecycle and telemetry envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["session_started", "telemetry", "session_paused", "session_reset", "session_closed", "episode_finished", "error"]
    session_id: str
    sequence: Annotated[int, Field(ge=0)]
    emitted_at: datetime
    source: Literal["live_pybullet"] = "live_pybullet"
    telemetry: TelemetrySample | None = None
    message: str | None = None


class ReplayDocument(BaseModel):
    """Portable measured recording used when the local service is offline."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_id: Literal[REPLAY_SCHEMA_ID] = Field(REPLAY_SCHEMA_ID, alias="schema")
    session: SessionRequest
    frames: tuple[TelemetrySample, ...]

    @model_validator(mode="after")
    def require_contiguous_frames(self) -> ReplayDocument:
        frames = [frame.frame for frame in self.frames]
        if frames != list(range(len(frames))):
            raise ValueError("replay frames must start at 0 and be contiguous")
        return self
