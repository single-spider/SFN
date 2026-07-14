"""Bounded local FastAPI service running actual PyBullet Panda sessions."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CameraConfig
from ..evaluation.evaluate_mfms import load_mfms_policy
from ..evaluation.evaluate_perception import _load_model
from ..evaluation.evaluate_sfms import _obs_to_state, load_sfms_policy
from ..models.controllers import SFSSController
from ..panda import PandaConfig, PandaPegInHoleInsertionEnv
from ..panda.native_vsn import PandaTopdownTemplateVSN
from ..training.train_mfms import make_mfms_history_state
from ..training.train_sfms import _require_torch
from .schema import ReplayDocument, SessionCommand, SessionEvent, SessionRequest, TelemetrySample

MAX_ACTIVE_SESSIONS = 2
SESSION_TTL = timedelta(minutes=5)
ROOT = Path(__file__).resolve().parents[2]


class SessionError(RuntimeError):
    """Raised for expired, missing, capacity-exhausted, or invalid sessions."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PolicyBundle:
    """Load inference models once per local showcase service process."""

    def __init__(self) -> None:
        torch, _ = _require_torch()
        self.torch = torch
        self.device = torch.device("cpu")
        self.segmentation = _load_model("segmentation", ROOT / "models" / "segmentation_panda_native_topdown_contrast.pt")
        self.segmentation.to(self.device).eval()
        self.sfms = load_sfms_policy(ROOT / "models" / "sfms_mesh_v2_rl_best_compatible.pt", str(self.device))
        self.mfms, self.mfms_history_len = load_mfms_policy(
            ROOT / "models" / "mfms_mesh_v2_teacher_compatible.pt", str(self.device)
        )
        self.sfss = SFSSController(max_xy_mm=2.0, max_yaw_deg=2.0, confidence_mode="ignore")

    def action(self, request: SessionRequest, obs: dict, vsn, history: list) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            if request.method == "sfss":
                rgb = torch.as_tensor(obs["rgb"][None], dtype=torch.float32, device=self.device)
                return self.sfss.act(vsn(rgb=rgb)).normalized.astype(np.float32)
            state = _obs_to_state(obs, vsn, "predicted", str(self.device))
            if request.method == "sfms":
                mean, _value = self.sfms(state)
            else:
                history.append(state)
                sequence = make_mfms_history_state(history, self.mfms_history_len, str(self.device))
                mean, _value, _hidden = self.mfms(sequence)
            return torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)


@dataclass
class _LiveSession:
    request: SessionRequest
    created_at: datetime
    queue: queue.Queue[SessionEvent] = field(default_factory=queue.Queue)
    pause_event: threading.Event = field(default_factory=threading.Event)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    sequence: int = 0
    recorded_frames: list[TelemetrySample] = field(default_factory=list)

    def emit(self, event: str, **kwargs: object) -> SessionEvent:
        item = SessionEvent(
            session_id=str(kwargs.pop("session_id")),
            event=event,
            sequence=self.sequence,
            emitted_at=_utc_now(),
            **kwargs,
        )
        self.sequence += 1
        if item.event == "telemetry" and item.telemetry is not None:
            self.recorded_frames.append(item.telemetry)
        self.queue.put(item)
        return item


class SessionManager:
    """Manage no more than two independent local PyBullet showcase sessions."""

    def __init__(self, *, bundle_factory: Callable[[], PolicyBundle] = PolicyBundle, now: Callable[[], datetime] = _utc_now) -> None:
        self._bundle_factory = bundle_factory
        self._bundle: PolicyBundle | None = None
        self._now = now
        self._sessions: dict[str, _LiveSession] = {}
        self._lock = threading.RLock()

    def _cleanup_locked(self) -> None:
        now = self._now()
        expired = [key for key, value in self._sessions.items() if now - value.created_at >= SESSION_TTL]
        for key in expired:
            self._sessions[key].stop_event.set()
            del self._sessions[key]

    def _get_bundle(self) -> PolicyBundle:
        if self._bundle is None:
            self._bundle = self._bundle_factory()
        return self._bundle

    def create(self, request: SessionRequest) -> SessionEvent:
        with self._lock:
            self._cleanup_locked()
            if len(self._sessions) >= MAX_ACTIVE_SESSIONS:
                raise SessionError(f"session capacity reached ({MAX_ACTIVE_SESSIONS})")
            session_id = uuid4().hex
            session = _LiveSession(request=request, created_at=self._now())
            self._sessions[session_id] = session
            return session.emit("session_started", session_id=session_id, message="PyBullet session ready")

    def command(self, session_id: str, command: SessionCommand) -> SessionEvent:
        with self._lock:
            self._cleanup_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("session not found or expired")
            if command.command == "close":
                session.stop_event.set()
                event = session.emit("session_closed", session_id=session_id, message="session closed")
                del self._sessions[session_id]
                return event
            if command.command == "pause":
                session.pause_event.set()
                return session.emit("session_paused", session_id=session_id, message="paused")
            if command.command == "reset":
                session.stop_event.set()
                session.pause_event.clear()
                session.thread = None
                return session.emit("session_reset", session_id=session_id, message="reset; start to run again")
            if session.thread is not None and session.thread.is_alive():
                session.pause_event.clear()
                return session.emit("session_started", session_id=session_id, message="already running")
            session.stop_event.clear()
            session.pause_event.clear()
            session.thread = threading.Thread(target=self._run_session, args=(session_id, session), daemon=True)
            session.thread.start()
            return session.emit("session_started", session_id=session_id, message="PyBullet run started")

    def _run_session(self, session_id: str, session: _LiveSession) -> None:
        env = None
        try:
            bundle = self._get_bundle()
            panda = PandaConfig(
                gui=False,
                execution_mode="dynamic",
                native_camera=True,
                camera_ignore_robot_occlusion=True,
                mesh_derived_alignment_z=True,
                camera_eye_offset_m=(0.0, 0.0, 0.20),
                camera_target_offset_m=(0.0, 0.0, 0.03),
            )
            camera = CameraConfig(crop_width=500, crop_height=400, fov_y_deg=35.0)
            phase = {"value": "ready"}
            action = {"value": None}
            frame = {"value": 0}

            def emit_state(state, _substep: int, _total: int) -> None:
                if env is None or env.scene is None:
                    return
                fingers = [
                    float(env.scene.p.getJointState(env.scene.ids.robot, index, physicsClientId=env.scene.client_id)[0])
                    for index in env.scene.config.finger_joint_indices
                ]
                sample = TelemetrySample(
                    frame=frame["value"],
                    phase=phase["value"],
                    joint_positions=tuple(float(x) for x in state.joint_positions) + tuple(fingers),
                    peg_pose=tuple(float(x) for x in [*state.peg_pos_world, *state.peg_quat_world]),
                    hole_pose=tuple(float(x) for x in [*state.hole_pos_world, *state.hole_quat_world]),
                    xy_error_mm=float(np.linalg.norm(state.pose_error_task[:2]) * 1000.0),
                    yaw_error_deg=abs(float(state.pose_error_task[2])),
                    action=None if action["value"] is None else tuple(float(x) for x in action["value"]),
                )
                frame["value"] += 1
                session.emit("telemetry", session_id=session_id, telemetry=sample)

            env = PandaPegInHoleInsertionEnv(
                shapes=[session.request.shape],
                panda_config=panda,
                camera_config=camera,
                motion_observer=emit_state,
                motion_observer_stride=8,
            )
            obs, _info = env.reset(seed=session.request.seed, options={"shape": session.request.shape})
            vsn = PandaTopdownTemplateVSN(
                session.request.shape,
                panda,
                camera.crop_width,
                camera.crop_height,
                camera.fov_y_deg,
                segmentation=bundle.segmentation,
            ).to(bundle.device).eval()
            bundle.sfss.reset()
            history: list = []
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated or session.stop_event.is_set()):
                while session.pause_event.is_set() and not session.stop_event.is_set():
                    time.sleep(0.05)
                if session.stop_event.is_set():
                    break
                phase["value"] = "alignment"
                action["value"] = bundle.action(session.request, obs, vsn, history)
                obs, _reward, terminated, truncated, info = env.step(action["value"])
                if info.get("insertion_attempted") or info.get("insertion_trace"):
                    phase["value"] = "insertion"
                final_state = env.scene.measure()
                fingers = [
                    float(env.scene.p.getJointState(env.scene.ids.robot, index, physicsClientId=env.scene.client_id)[0])
                    for index in env.scene.config.finger_joint_indices
                ]
                final = TelemetrySample(
                    frame=frame["value"],
                    phase="complete" if info.get("insertion_success") else ("failed" if terminated or truncated else phase["value"]),
                    joint_positions=tuple(float(x) for x in final_state.joint_positions) + tuple(fingers),
                    peg_pose=tuple(float(x) for x in [*final_state.peg_pos_world, *final_state.peg_quat_world]),
                    hole_pose=tuple(float(x) for x in [*final_state.hole_pos_world, *final_state.hole_quat_world]),
                    xy_error_mm=float(info.get("xy_error_mm", np.linalg.norm(final_state.pose_error_task[:2]) * 1000.0)),
                    yaw_error_deg=float(info.get("yaw_error_deg", abs(final_state.pose_error_task[2]))),
                    action=tuple(float(x) for x in action["value"]),
                    insertion_depth_mm=info.get("insertion_depth_mm"),
                    contact_count=int(info.get("contact_count", 0)),
                    max_contact_force=float(info.get("max_contact_force", 0.0)),
                    tracking_error_mm=float(info.get("tracking_error_mm", 0.0)),
                    terminated=bool(terminated or truncated),
                    success=bool(info.get("insertion_success", False)),
                    reason=info.get("termination_reason"),
                )
                frame["value"] += 1
                session.emit("telemetry", session_id=session_id, telemetry=final)
            session.emit(
                "episode_finished",
                session_id=session_id,
                message="stopped" if session.stop_event.is_set() else str(info.get("termination_reason", "finished")),
            )
        except Exception as error:  # pragma: no cover - defensive service boundary
            session.emit("error", session_id=session_id, message=f"{type(error).__name__}: {error}")
        finally:
            if env is not None:
                env.close()

    def take_event(self, session_id: str, timeout_s: float = 0.25) -> SessionEvent | None:
        with self._lock:
            self._cleanup_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("session not found or expired")
        try:
            return session.queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def recording(self, session_id: str) -> ReplayDocument:
        """Return the measured telemetry captured for a completed local run."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("session not found or expired")
            if session.thread is not None and session.thread.is_alive():
                raise SessionError("recording is not complete")
            return ReplayDocument(session=session.request, frames=tuple(session.recorded_frames))

    def health(self) -> dict[str, object]:
        with self._lock:
            self._cleanup_locked()
            return {
                "status": "ok",
                "adapter": "live_pybullet",
                "live_pybullet": True,
                "active_sessions": len(self._sessions),
                "max_sessions": MAX_ACTIVE_SESSIONS,
                "session_ttl_seconds": int(SESSION_TTL.total_seconds()),
                "methods": ["sfss", "sfms", "mfms"],
            }


def create_app(manager: SessionManager | None = None):
    """Create the local-only FastAPI app without exposing arbitrary robot control."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
    except ModuleNotFoundError as error:  # pragma: no cover
        raise RuntimeError("Showcase service requires optional dependencies: fastapi and uvicorn") from error

    session_manager = manager or SessionManager()
    app = FastAPI(title="SFN Showcase Live Panda", version="2")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type"],
    )

    @app.get("/v1/health")
    def health() -> dict[str, object]:
        return session_manager.health()

    @app.post("/v1/sessions", response_model=SessionEvent, status_code=201)
    def create_session(request: SessionRequest) -> SessionEvent:
        try:
            return session_manager.create(request)
        except SessionError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error

    @app.post("/v1/sessions/{session_id}/commands", response_model=SessionEvent)
    def execute_command(session_id: str, command: SessionCommand) -> SessionEvent:
        try:
            return session_manager.command(session_id, command)
        except SessionError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.websocket("/v1/sessions/{session_id}/stream")
    async def websocket_session(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            while True:
                event = await asyncio.to_thread(session_manager.take_event, session_id, 0.25)
                if event is not None:
                    await websocket.send_json(event.model_dump(mode="json"))
                    if event.event in {"session_closed", "episode_finished", "error"}:
                        return
        except (SessionError, WebSocketDisconnect):
            return

    return app
