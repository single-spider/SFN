"""Deterministic JSON replay loading for the showcase telemetry contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import ReplayDocument


def load_replay(path: str | Path) -> ReplayDocument:
    """Load and validate one replay without executing or importing its contents."""
    replay_path = Path(path)
    try:
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid showcase replay JSON: {replay_path}") from error
    return ReplayDocument.model_validate(payload)


def replay_fingerprint(replay: ReplayDocument) -> str:
    """Return a stable fingerprint for a validated replay document."""
    payload = json.dumps(replay.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
