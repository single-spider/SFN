import numpy as np
from sfn.training.vector_env import SynchronousEnvRunner


class DummyEnv:
    def __init__(self):
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed, options):
        self.step_count = 0
        return {"seed": seed, "shape": options["shape"], "terminal": False}, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count == 2
        return {"terminal": done}, float(action[0]), False, done, {"step": self.step_count}

    def close(self):
        self.closed = True


def test_synchronous_runner_seed_boundary_and_cleanup():
    runner = SynchronousEnvRunner([DummyEnv, DummyEnv], seed=10, shapes=["a", "b"])
    observations = runner.reset()
    assert [(row["seed"], row["shape"]) for row in observations] == [(10, "a"), (11, "b")]
    first = runner.step([np.array([1.0]), np.array([2.0])])
    assert not first.truncated.any()
    second = runner.step([np.array([1.0]), np.array([2.0])])
    assert second.truncated.tolist() == [True, True]
    assert all(row == {"terminal": True} for row in second.boundary_observations)
    assert [(row["seed"], row["shape"]) for row in second.observations] == [(12, "a"), (13, "b")]
    envs = list(runner.envs)
    runner.close()
    assert all(env.closed for env in envs)


def test_synchronous_runner_rejects_wrong_action_count():
    with SynchronousEnvRunner([DummyEnv, DummyEnv], seed=1, shapes=["a"]) as runner:
        runner.reset()
        try:
            runner.step([np.array([0.0])])
        except ValueError as exc:
            assert "expected 2 actions" in str(exc)
        else:
            raise AssertionError("wrong action count was accepted")
