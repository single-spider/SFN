"""Bounded, synthetic live-service primitives for the SFN showcase.

This package deliberately has no dependency on Panda or PyBullet.  The first
adapter is synthetic so that a UI can be integrated and tested safely before a
real simulator is wired in.
"""

from .replay import load_replay, replay_fingerprint
from .service import SessionManager, create_app

__all__ = ["SessionManager", "create_app", "load_replay", "replay_fingerprint"]
