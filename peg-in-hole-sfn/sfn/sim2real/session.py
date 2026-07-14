"""Dry-run/armed/reacquisition lifecycle around the command safety gate."""

from __future__ import annotations

from enum import StrEnum

from .interfaces import CartesianCommand, CommandSink, RobotState
from .safety import CommandSafetyGate, SafetyDecision, SafetyDisposition


class SessionState(StrEnum):
    DRY_RUN = "dry_run"
    ARMED = "armed"
    REACQUIRE = "reacquire"
    STOPPED = "stopped"


class GuardedCommandSession:
    def __init__(self, sink: CommandSink, gate: CommandSafetyGate, *, dry_run: bool = True) -> None:
        self.sink = sink
        self.gate = gate
        self.state = SessionState.DRY_RUN if dry_run else SessionState.ARMED

    def arm(self) -> None:
        if self.state is SessionState.STOPPED:
            raise RuntimeError("a stopped session cannot be armed")
        self.state = SessionState.ARMED

    def mark_reacquired(self, *, reset_motion_budget: bool = False) -> None:
        if self.state is not SessionState.REACQUIRE:
            raise RuntimeError("session is not awaiting reacquisition")
        if reset_motion_budget:
            self.gate.reset()
        self.state = SessionState.ARMED

    def stop(self, reason: str = "operator_stop") -> None:
        self.sink.stop(reason)
        self.state = SessionState.STOPPED

    def submit(self, command: CartesianCommand, state: RobotState, **observation: float) -> SafetyDecision:
        if self.state is SessionState.STOPPED:
            return SafetyDecision(SafetyDisposition.REJECT, ("session_stopped",))
        if self.state is SessionState.REACQUIRE:
            return SafetyDecision(SafetyDisposition.REACQUIRE, ("reacquisition_required",))
        decision = self.gate.evaluate(command, state, **observation)
        if decision.disposition is SafetyDisposition.REACQUIRE:
            self.state = SessionState.REACQUIRE
        elif decision.accepted:
            if self.state is SessionState.ARMED:
                self.sink.send(command)
            self.gate.commit(command)
        return decision
