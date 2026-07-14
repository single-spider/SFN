from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneState:
    shape: str
    pose_error: np.ndarray
    step_count: int = 0
