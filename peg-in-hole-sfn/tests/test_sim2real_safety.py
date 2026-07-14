from __future__ import annotations

import numpy as np
from sfn.sim2real import (
    CartesianCommand,
    CommandSafetyGate,
    GuardedCommandSession,
    RecordingCommandSink,
    RobotState,
    SafetyDisposition,
    SafetyLimits,
    SessionState,
)


def command(x: float, *, at: float = 1.0) -> CartesianCommand:
    return CartesianCommand(np.array([x, 0.0, 0.0]), issued_at=at)


def test_gate_accepts_and_accounts_for_safe_command():
    gate = CommandSafetyGate(SafetyLimits(workspace_min_m=(-1, -1, -1), workspace_max_m=(1, 1, 1)))
    state = RobotState(np.zeros(3))
    decision = gate.evaluate(command(0.001), state, confidence=0.9, frame_timestamp=0.95, now=1.0)
    assert decision.accepted
    gate.commit(command(0.001))
    assert gate.cumulative_translation_m == 0.001


def test_confidence_and_stale_frames_request_reacquisition():
    gate = CommandSafetyGate()
    state = RobotState(np.zeros(3))
    assert (
        gate.evaluate(command(0), state, confidence=0.1, frame_timestamp=1, now=1).disposition
        is SafetyDisposition.REACQUIRE
    )
    decision = gate.evaluate(command(0), state, confidence=1, frame_timestamp=0, now=1)
    assert decision.reasons == ("stale_frame",)


def test_action_workspace_rate_and_cumulative_limits_reject():
    limits = SafetyLimits(
        max_translation_m=0.01,
        max_translation_rate_m_s=0.01,
        workspace_min_m=(-0.02, -1, -1),
        workspace_max_m=(0.02, 1, 1),
        max_cumulative_translation_m=0.006,
    )
    state = RobotState(np.array([0.019, 0, 0]))
    gate = CommandSafetyGate(limits)
    first = command(0.005, at=1)
    decision = gate.evaluate(first, state, confidence=1, frame_timestamp=1, now=1)
    assert "workspace_limit" in decision.reasons
    gate.commit(first)
    second = command(-0.005, at=1.1)
    decision = gate.evaluate(second, RobotState(np.zeros(3)), confidence=1, frame_timestamp=1.1, now=1.1)
    assert {"translation_rate_limit", "cumulative_translation_limit"} <= set(decision.reasons)
    oversized = gate.evaluate(command(0.02, at=2), state, confidence=1, frame_timestamp=2, now=2)
    assert "translation_limit" in oversized.reasons


def test_dry_run_reacquire_arm_and_stop_lifecycle():
    sink = RecordingCommandSink()
    session = GuardedCommandSession(sink, CommandSafetyGate(), dry_run=True)
    state = RobotState(np.zeros(3))
    assert session.submit(command(0), state, confidence=1, frame_timestamp=1, now=1).accepted
    assert sink.commands == []
    session.arm()
    decision = session.submit(command(0, at=2), state, confidence=0, frame_timestamp=2, now=2)
    assert decision.disposition is SafetyDisposition.REACQUIRE
    assert session.state is SessionState.REACQUIRE
    session.mark_reacquired()
    session.submit(command(0, at=3), state, confidence=1, frame_timestamp=3, now=3)
    assert len(sink.commands) == 1
    session.stop("test")
    assert sink.stop_reasons == ["test"]
