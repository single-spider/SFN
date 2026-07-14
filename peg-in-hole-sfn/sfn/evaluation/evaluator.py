from __future__ import annotations

from ..config import EnvironmentConfig, InsertionConfig
from ..envs import PegInHoleAlignmentEnv, PegInHoleInsertionEnv
from ..models.controllers import OracleController


def evaluate_oracle(
    shapes: list[str] | None = None,
    episodes_per_shape: int = 10,
    seed: int = 1,
    task: str = "alignment",
    env_config: EnvironmentConfig | None = None,
    insertion_config: InsertionConfig | None = None,
) -> list[dict]:
    env_cls = PegInHoleInsertionEnv if task == "insertion" else PegInHoleAlignmentEnv
    kwargs = {"shapes": shapes, "seed": seed, "env_config": env_config}
    if task == "insertion":
        kwargs["insertion_config"] = insertion_config
    env = env_cls(**kwargs)
    ctrl = OracleController(env.config.max_action_xy_mm, env.config.max_action_yaw_deg)
    records = []
    try:
        global_episode = 0
        for shape in shapes or env.shapes:
            for ep in range(episodes_per_shape):
                # Use a globally unique seed per episode.  Reusing ``seed + ep``
                # inside every shape silently gives every shape the same
                # initial pose schedule, which makes aggregate reports look
                # suspiciously identical across splits.
                obs, info = env.reset(seed=seed + global_episode, options={"shape": shape})
                total = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = ctrl.act_from_pose_error(obs["pose_error"])
                    obs, reward, terminated, truncated, info = env.step(action.normalized)
                    total += reward
                records.append(
                    {
                        "shape": shape,
                        "episode": ep,
                        "task": task,
                        "success": bool(info["success"]),
                        "steps": info["step"],
                        "reward": total,
                        "final_xy_error_mm": info["xy_error_mm"],
                        "final_yaw_error_deg": info["yaw_error_deg"],
                        "termination_reason": info.get(
                            "termination_reason", "alignment_success" if info.get("success") else "unknown"
                        ),
                        "insertion_depth_mm": info.get("insertion_depth_mm"),
                        "collision_failure": info.get("collision_failure"),
                    }
                )
                global_episode += 1
    finally:
        env.close()
    return records
