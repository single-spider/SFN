"""Deterministic, robot-agnostic command gating for real-world trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .interfaces import CartesianCommand, RobotState


class SafetyDisposition(StrEnum):
    ACCEPT = "accept"
    REACQUIRE = "reacquire"
    REJECT = "reject"


@dataclass(frozen=True)
class SafetyLimits:
    min_confidence: float = 0.6
    max_frame_age_s: float = 0.25
    max_translation_m: float = 0.005
    max_rotation_rad: float = 0.05
    max_translation_rate_m_s: float = 0.025
    max_rotation_rate_rad_s: float = 0.25
    workspace_min_m: tuple[float, float, float] = (-np.inf, -np.inf, -np.inf)
    workspace_max_m: tuple[float, float, float] = (np.inf, np.inf, np.inf)
    max_cumulative_translation_m: float = 0.05
    max_cumulative_rotation_rad: float = 0.5


@dataclass(frozen=True)
class SafetyDecision:
    disposition: SafetyDisposition
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.disposition is SafetyDisposition.ACCEPT


class CommandSafetyGate:
    """Stateful gate. Accepted motion contributes to rate and travel budgets."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits if limits is not None else SafetyLimits()
        self.last_command: CartesianCommand | None = None
        self.cumulative_translation_m = 0.0
        self.cumulative_rotation_rad = 0.0

    def reset(self) -> None:
        self.last_command = None
        self.cumulative_translation_m = 0.0
        self.cumulative_rotation_rad = 0.0

    def evaluate(
        self,
        command: CartesianCommand,
        state: RobotState,
        *,
        confidence: float,
        frame_timestamp: float,
        now: float,
    ) -> SafetyDecision:
        if not np.isfinite(confidence) or confidence < self.limits.min_confidence:
            return SafetyDecision(SafetyDisposition.REACQUIRE, ("low_confidence",))
        if (
            not np.isfinite(frame_timestamp)
            or now - frame_timestamp > self.limits.max_frame_age_s
            or frame_timestamp > now
        ):
            return SafetyDecision(SafetyDisposition.REACQUIRE, ("stale_frame",))

        reasons: list[str] = []
        translation = command.translation_m
        rotation = command.rotation_rad
        if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
            reasons.append("non_finite_action")
        else:
            translation_norm = float(np.linalg.norm(translation))
            rotation_norm = float(np.linalg.norm(rotation))
            if translation_norm > self.limits.max_translation_m:
                reasons.append("translation_limit")
            if rotation_norm > self.limits.max_rotation_rad:
                reasons.append("rotation_limit")
            projected = state.position_m + translation
            if np.any(projected < np.asarray(self.limits.workspace_min_m)) or np.any(
                projected > np.asarray(self.limits.workspace_max_m)
            ):
                reasons.append("workspace_limit")
            if self.cumulative_translation_m + translation_norm > self.limits.max_cumulative_translation_m:
                reasons.append("cumulative_translation_limit")
            if self.cumulative_rotation_rad + rotation_norm > self.limits.max_cumulative_rotation_rad:
                reasons.append("cumulative_rotation_limit")
            if self.last_command is not None:
                dt = command.issued_at - self.last_command.issued_at
                if dt <= 0:
                    reasons.append("non_monotonic_command_time")
                else:
                    if (
                        float(np.linalg.norm(translation - self.last_command.translation_m)) / dt
                        > self.limits.max_translation_rate_m_s
                    ):
                        reasons.append("translation_rate_limit")
                    if (
                        float(np.linalg.norm(rotation - self.last_command.rotation_rad)) / dt
                        > self.limits.max_rotation_rate_rad_s
                    ):
                        reasons.append("rotation_rate_limit")

        return (
            SafetyDecision(SafetyDisposition.REJECT, tuple(reasons))
            if reasons
            else SafetyDecision(SafetyDisposition.ACCEPT)
        )

    def commit(self, command: CartesianCommand) -> None:
        """Account for a command only after the downstream sink accepts it."""
        self.cumulative_translation_m += float(np.linalg.norm(command.translation_m))
        self.cumulative_rotation_rad += float(np.linalg.norm(command.rotation_rad))
        self.last_command = command
