"""Reproducible seeding helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int
    numpy_generator: np.random.Generator


def seed_everything(seed: int, deterministic_torch: bool = True) -> SeedState:
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return SeedState(seed, np.random.default_rng(seed))


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))
