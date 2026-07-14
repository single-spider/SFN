import pytest

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env
from sfn.config import CameraConfig
from sfn.envs import PegInHoleAlignmentEnv, PegInHoleInsertionEnv


@pytest.mark.parametrize("env_class", [PegInHoleAlignmentEnv, PegInHoleInsertionEnv])
@pytest.mark.parametrize("backend", ["toy_direct", "mesh_orthographic"])
def test_standalone_gymnasium_contracts(env_class, backend):
    env = env_class(
        shapes=["square-square"] if backend == "mesh_orthographic" else ["synthetic-square"],
        camera_config=CameraConfig(renderer_backend=backend),
    )
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_panda_gymnasium_contracts():
    pytest.importorskip("pybullet")
    from sfn.panda import PandaConfig
    from sfn.panda.panda_alignment_env import PandaPegInHoleAlignmentEnv
    from sfn.panda.panda_insertion_env import PandaPegInHoleInsertionEnv

    for env_class in (PandaPegInHoleAlignmentEnv, PandaPegInHoleInsertionEnv):
        env = env_class(
            shapes=["square-square"],
            panda_config=PandaConfig(execution_mode="kinematic", mesh_derived_alignment_z=True),
        )
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
