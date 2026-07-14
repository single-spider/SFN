from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sfn.constants import ALL_EXPECTED_SHAPES
from sfn.showcase.schema import SessionCommand, SessionRequest
from sfn.showcase.service import MAX_ACTIVE_SESSIONS, SESSION_TTL, SessionError, SessionManager, create_app


def test_session_manager_is_bounded_without_loading_models() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    manager = SessionManager(now=lambda: now)
    request = SessionRequest(shape=ALL_EXPECTED_SHAPES[0], method="sfms")
    events = [manager.create(request) for _ in range(MAX_ACTIVE_SESSIONS)]
    assert all(event.source == "live_pybullet" for event in events)
    assert manager.health()["live_pybullet"] is True
    with pytest.raises(SessionError, match="capacity"):
        manager.create(request)


def test_session_expiry_releases_capacity_without_running_robot() -> None:
    clock = [datetime(2026, 1, 1, tzinfo=UTC)]
    manager = SessionManager(now=lambda: clock[0])
    started = manager.create(SessionRequest(shape=ALL_EXPECTED_SHAPES[0], seed=7))
    clock[0] += SESSION_TTL + timedelta(seconds=1)
    with pytest.raises(SessionError, match="expired"):
        manager.command(started.session_id, SessionCommand(command="reset"))
    assert manager.health()["active_sessions"] == 0


def test_models_reject_unallowlisted_controls() -> None:
    with pytest.raises(ValueError, match="shape must"):
        SessionRequest(shape="../../anything")
    with pytest.raises(ValueError, match="method must"):
        SessionRequest(shape=ALL_EXPECTED_SHAPES[0], method="shell")
    with pytest.raises(ValueError):
        SessionCommand.model_validate({"command": "step"})


def test_health_endpoint_when_fastapi_is_installed() -> None:
    from fastapi.testclient import TestClient

    response = TestClient(create_app()).get("/v1/health")
    assert response.status_code == 200
    assert response.json()["live_pybullet"] is True
