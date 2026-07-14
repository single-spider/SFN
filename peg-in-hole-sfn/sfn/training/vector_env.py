"""Small synchronous vector runner with explicit episode-boundary semantics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VectorStep:
    observations: list[dict]
    boundary_observations: list[dict | None]
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict]


class SynchronousEnvRunner:
    """Step independent Gymnasium environments in a deterministic fixed order.

    Unlike an implicit autoreset vector wrapper, this runner retains the final
    observation separately so truncations can be bootstrapped correctly.
    """

    def __init__(self, factories: Sequence[Callable[[], Any]], *, seed: int, shapes: Sequence[str]) -> None:
        if not factories:
            raise ValueError("at least one environment factory is required")
        if not shapes:
            raise ValueError("at least one shape is required")
        self.envs = [factory() for factory in factories]
        self.seed = int(seed)
        self.shapes = list(shapes)
        self.episode_counts = [0] * len(self.envs)
        self.current_observations: list[dict] = []

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    def _episode_seed(self, index: int) -> int:
        return self.seed + index + self.episode_counts[index] * self.num_envs

    def _reset_one(self, index: int) -> dict:
        episode = self.episode_counts[index]
        shape = self.shapes[(index + episode * self.num_envs) % len(self.shapes)]
        observation, _info = self.envs[index].reset(
            seed=self._episode_seed(index),
            options={"shape": shape},
        )
        return observation

    def reset(self) -> list[dict]:
        self.episode_counts = [0] * self.num_envs
        self.current_observations = [self._reset_one(index) for index in range(self.num_envs)]
        return self.current_observations

    def step(self, actions: Sequence[np.ndarray]) -> VectorStep:
        if len(actions) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} actions, got {len(actions)}")
        observations: list[dict] = []
        boundaries: list[dict | None] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict] = []
        for index, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            observation, reward, term, trunc, info = env.step(action)
            rewards[index] = float(reward)
            terminated[index] = bool(term)
            truncated[index] = bool(trunc)
            boundaries.append(observation if term or trunc else None)
            if term or trunc:
                self.episode_counts[index] += 1
                observation = self._reset_one(index)
            observations.append(observation)
            infos.append(info)
        self.current_observations = observations
        return VectorStep(observations, boundaries, rewards, terminated, truncated, infos)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def state_dict(self) -> dict:
        environments = []
        for env in self.envs:
            state = getattr(env, "state", None)
            if state is None:
                raise TypeError(f"{type(env).__name__} does not expose resumable state")
            environments.append(
                {
                    "shape": str(state.shape),
                    "pose_error": np.asarray(state.pose_error, dtype=np.float32).tolist(),
                    "step_count": int(state.step_count),
                    "seed": int(env._seed),
                    "rng_state": env.rng.bit_generator.state,
                }
            )
        observations = [
            {key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value for key, value in row.items()}
            for row in self.current_observations
        ]
        return {"episode_counts": list(self.episode_counts), "environments": environments, "observations": observations}

    def load_state_dict(self, state: dict) -> list[dict]:
        rows = state.get("environments", [])
        if len(rows) != self.num_envs:
            raise ValueError("vector runner state has incompatible environment count")
        self.episode_counts = [int(value) for value in state["episode_counts"]]
        observations = []
        for env, row in zip(self.envs, rows, strict=True):
            observation, _info = env.reset(
                seed=int(row["seed"]),
                options={"shape": row["shape"], "pose_error": row["pose_error"], "nontrivial": False},
            )
            env.state.step_count = int(row["step_count"])
            env.rng.bit_generator.state = row["rng_state"]
            observations.append(
                {
                    key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                    for key, value in state["observations"][len(observations)].items()
                }
            )
        self.current_observations = observations
        return observations

    def __enter__(self) -> SynchronousEnvRunner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
