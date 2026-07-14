def hide_pose_error(obs: dict) -> dict:
    return {k: v for k, v in obs.items() if k != "pose_error"}
