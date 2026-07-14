"""Gym/Gymnasium compatibility with a tiny fallback."""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym

    spaces = gym.spaces
except ModuleNotFoundError:
    try:
        import gym

        spaces = gym.spaces
    except ModuleNotFoundError:

        class Env:
            def reset(self, *, seed=None, options=None):
                return None, {}

            def step(self, action):
                raise NotImplementedError

            def close(self):
                pass

        class Box:
            def __init__(self, low, high, shape, dtype):
                self.shape = tuple(shape)
                self.dtype = np.dtype(dtype)
                self.low = (
                    np.full(self.shape, low, dtype=self.dtype)
                    if np.isscalar(low)
                    else np.asarray(low, dtype=self.dtype)
                )
                self.high = (
                    np.full(self.shape, high, dtype=self.dtype)
                    if np.isscalar(high)
                    else np.asarray(high, dtype=self.dtype)
                )

            def sample(self):
                if np.issubdtype(self.dtype, np.integer):
                    return np.random.randint(self.low, self.high + 1, size=self.shape, dtype=self.dtype)
                return np.random.uniform(self.low, self.high, size=self.shape).astype(self.dtype)

            def contains(self, x):
                arr = np.asarray(x)
                return (
                    arr.shape == self.shape
                    and np.can_cast(arr.dtype, self.dtype, casting="same_kind")
                    and np.all(arr >= self.low)
                    and np.all(arr <= self.high)
                )

        class Dict:
            def __init__(self, spaces_dict):
                self.spaces = dict(spaces_dict)

            def sample(self):
                return {k: v.sample() for k, v in self.spaces.items()}

            def contains(self, x):
                return (
                    isinstance(x, dict)
                    and set(x) == set(self.spaces)
                    and all(self.spaces[k].contains(x[k]) for k in self.spaces)
                )

        class _Spaces:
            Box = Box
            Dict = Dict

        class _Gym:
            Env = Env

        gym = _Gym()
        spaces = _Spaces()
