"""Software-only building blocks for guarded sim-to-real evaluation."""

from .active_learning import select_active_learning_frames, write_pseudolabel_review_manifest
from .calibration import CameraCalibration, load_camera_calibration, save_camera_calibration
from .interfaces import CartesianCommand, CommandSink, RecordingCommandSink, RobotState
from .safety import CommandSafetyGate, SafetyDecision, SafetyDisposition, SafetyLimits
from .session import GuardedCommandSession, SessionState

__all__ = [
    "CartesianCommand",
    "CommandSink",
    "RecordingCommandSink",
    "RobotState",
    "CommandSafetyGate",
    "SafetyDecision",
    "SafetyDisposition",
    "SafetyLimits",
    "GuardedCommandSession",
    "SessionState",
    "CameraCalibration",
    "load_camera_calibration",
    "save_camera_calibration",
    "select_active_learning_frames",
    "write_pseudolabel_review_manifest",
]
