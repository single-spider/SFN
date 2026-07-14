"""Single-frame multi-step (SFMS) A2C training utilities.

This module intentionally does not use ground-truth pose error as policy input.
The policy observes only the canonical VSN state:

    flatten(position_prob) + orientation_prob == 441 + 11 = 452 values.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CameraConfig, EnvironmentConfig
from ..envs import PegInHoleAlignmentEnv
from ..models.controllers import SFSSController
from ..models.vsn import VirtualSensorNetwork
from .common import file_sha256, load_checkpoint_cpu, make_checkpoint, run_metadata, save_checkpoint
from .rl_utils import compute_gae, tanh_normal_sample
from .vector_env import SynchronousEnvRunner


def _require_torch():
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without torch
        raise SystemExit("PyTorch is required for SFMS training.") from exc
    return torch, nn


@dataclass
class SFMSTrainConfig:
    updates: int = 10
    rollout_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    anchor_imitation_coef: float = 0.0
    grad_norm_clip: float = 0.5
    min_log_std: float = -4.0
    max_log_std: float = 0.5
    eval_every: int = 0
    eval_episodes_per_shape: int = 3
    eval_seed: int | None = None
    num_envs: int = 1
    checkpoint_every: int = 0
    log_jsonl: str | None = None
    seed: int = 1
    mask_source: str = "ground_truth"
    device: str = "cpu"


@dataclass
class SFMSTeacherPretrainConfig:
    samples: int = 4096
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 1
    mask_source: str = "ground_truth"
    device: str = "cpu"
    confidence_mode: str = "ignore"


def make_sfms_state(vsn_output: Any):
    """Return the canonical 452-value policy state from a VSN output."""
    torch, _ = _require_torch()
    pos = vsn_output.position_prob.flatten(1)
    ori = vsn_output.orientation_prob
    state = torch.cat([pos, ori], dim=1)
    if state.shape[1] != 452:
        raise ValueError(f"Expected 452-value SFMS state, got {tuple(state.shape)}")
    return state


class SFMSActorCritic(_require_torch()[1].Module):
    """Small A2C actor-critic matching the technical spec baseline."""

    def __init__(self, input_dim: int = 452, hidden1: int = 256, hidden2: int = 128, action_dim: int = 3):
        torch, nn = _require_torch()
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden2, action_dim)
        self.critic = nn.Linear(hidden2, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        torch, _ = _require_torch()
        h = self.body(x)
        mean = torch.tanh(self.actor_mean(h))
        value = self.critic(h).squeeze(-1)
        return mean, value

    def distribution(self, x):
        torch, _ = _require_torch()
        mean, value = self(x)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    def sample_action(self, x, deterministic: bool = False):
        h = self.body(x)
        raw_mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        action, log_prob, entropy = tanh_normal_sample(raw_mean, self.log_std, deterministic=deterministic)
        return action, log_prob, entropy, value


def _obs_to_state(obs: dict, vsn: VirtualSensorNetwork, mask_source: str, device: str):
    return _observations_to_state([obs], vsn, mask_source, device)


def _observations_to_state(observations: list[dict], vsn: VirtualSensorNetwork, mask_source: str, device: str):
    torch, _ = _require_torch()
    with torch.no_grad():
        if mask_source == "ground_truth":
            mask = torch.as_tensor(np.stack([obs["mask"] for obs in observations]), dtype=torch.long, device=device)
            out = vsn(mask=mask)
        elif mask_source == "predicted":
            rgb = torch.as_tensor(np.stack([obs["rgb"] for obs in observations]), dtype=torch.float32, device=device)
            out = vsn(rgb=rgb)
        else:
            raise ValueError("mask_source must be ground_truth or predicted")
        return make_sfms_state(out).detach()


def _deterministic_eval_model(
    model: Any,
    vsn: VirtualSensorNetwork,
    mask_source: str,
    device: str,
    shapes: list[str],
    env_config: EnvironmentConfig | None,
    camera_config: CameraConfig | None,
    seed: int,
    episodes_per_shape: int,
) -> dict[str, float]:
    """Evaluate an in-memory SFMS model using actor means.

    This stays inside the training module so long runs can select a best
    checkpoint without writing and reloading a policy at every interval.
    """
    torch, _ = _require_torch()
    env = PegInHoleAlignmentEnv(shapes=shapes, seed=seed, env_config=env_config, camera_config=camera_config)
    records: list[dict[str, float | bool]] = []
    was_training = bool(model.training)
    model.eval()
    try:
        global_episode = 0
        for shape in env.shapes:
            for _ep in range(episodes_per_shape):
                obs, info = env.reset(seed=seed + global_episode, options={"shape": shape})
                terminated = truncated = False
                while not (terminated or truncated):
                    with torch.no_grad():
                        state = _obs_to_state(obs, vsn, mask_source, device)
                        mean, _value = model(state)
                        action = torch.clamp(mean, -1.0, 1.0)[0].detach().cpu().numpy().astype(np.float32)
                    obs, _reward, terminated, truncated, info = env.step(action)
                records.append(
                    {
                        "success": bool(info["success"]),
                        "steps": float(info["step"]),
                        "final_xy_error_mm": float(info["xy_error_mm"]),
                        "final_yaw_error_deg": float(info["yaw_error_deg"]),
                    }
                )
                global_episode += 1
    finally:
        env.close()
        if was_training:
            model.train()
    if not records:
        return {"success_rate": 0.0, "mean_steps": 0.0, "mean_final_xy_error_mm": 0.0, "mean_final_yaw_error_deg": 0.0}
    return {
        "success_rate": float(np.mean([r["success"] for r in records])),
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "mean_final_xy_error_mm": float(np.mean([r["final_xy_error_mm"] for r in records])),
        "mean_final_yaw_error_deg": float(np.mean([r["final_yaw_error_deg"] for r in records])),
    }


def random_policy_smoke(
    episodes: int = 2,
    shapes: list[str] | None = None,
    seed: int = 1,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
) -> dict:
    """Run a random continuous-action policy and return finite smoke metrics."""
    env = PegInHoleAlignmentEnv(
        shapes=shapes or ["synthetic-square"], seed=seed, env_config=env_config, camera_config=camera_config
    )
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    successes = 0
    try:
        for ep in range(episodes):
            _obs, info = env.reset(seed=seed + ep, options={"shape": env.shapes[ep % len(env.shapes)]})
            done = False
            total = 0.0
            while not done:
                action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                _obs, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            rewards.append(total)
            successes += int(info["success"])
    finally:
        env.close()
    return {
        "episodes": episodes,
        "success_rate": successes / max(1, episodes),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def train_sfms(
    out: str | Path,
    config: SFMSTrainConfig | None = None,
    shapes: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
    vsn: VirtualSensorNetwork | None = None,
    initial_policy_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    eval_shapes: list[str] | None = None,
    best_out: str | Path | None = None,
) -> dict:
    """Train a compact A2C SFMS policy and save a portable checkpoint."""
    torch, _ = _require_torch()
    cfg = config or SFMSTrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")

    if cfg.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    train_shapes = shapes or ["synthetic-square"]
    factories = [
        lambda index=index: PegInHoleAlignmentEnv(
            shapes=train_shapes,
            seed=cfg.seed + index,
            env_config=env_config,
            camera_config=camera_config,
        )
        for index in range(cfg.num_envs)
    ]
    runner = SynchronousEnvRunner(factories, seed=cfg.seed, shapes=train_shapes)
    if vsn is None:
        vsn = VirtualSensorNetwork.from_checkpoints(
            segmentation_path if cfg.mask_source == "predicted" else None,
            position_path,
            orientation_path,
        )
    vsn.to(device).eval()

    model = SFMSActorCritic().to(device)
    if initial_policy_path is not None and resume_path is not None:
        raise ValueError("initial_policy_path and resume_path are mutually exclusive")
    resume_ckpt = None
    if resume_path is not None:
        resume_ckpt = load_checkpoint_cpu(resume_path)
        model.load_state_dict(resume_ckpt["model_state_dict"])
    elif initial_policy_path is not None:
        ckpt = load_checkpoint_cpu(initial_policy_path)
        model.load_state_dict(ckpt["model_state_dict"])
    anchor_model = None
    if initial_policy_path is not None and cfg.anchor_imitation_coef > 0:
        anchor_model = copy.deepcopy(model).to(device).eval()
        for param in anchor_model.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.body.parameters()) + list(model.actor_mean.parameters()) + [model.log_std],
                "lr": cfg.actor_lr,
            },
            {"params": model.critic.parameters(), "lr": cfg.critic_lr},
        ]
    )
    if resume_ckpt is not None and resume_ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    observations = runner.reset()
    metrics: dict[str, Any] = {}
    best_metrics: dict[str, Any] | None = None
    global_step = int(resume_ckpt.get("global_step", 0)) if resume_ckpt else 0
    start_update = int(resume_ckpt.get("epoch", 0)) if resume_ckpt else 0
    if resume_ckpt and resume_ckpt.get("training_state"):
        saved_state = resume_ckpt["training_state"]
        observations = runner.load_state_dict(saved_state["runner"])
        torch.set_rng_state(saved_state["torch_rng_state"])
        numpy_state = list(saved_state["numpy_rng_state"])
        if torch.is_tensor(numpy_state[1]):
            numpy_state[1] = numpy_state[1].cpu().numpy()
        np.random.set_state(tuple(numpy_state))
        if torch.cuda.is_available() and saved_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved_state["cuda_rng_state"])
    run = run_metadata(cfg.seed)
    compatibility = {
        "renderer_backend": getattr(camera_config, "renderer_backend", None),
        "mask_source": cfg.mask_source,
        "segmentation_sha256": file_sha256(segmentation_path),
        "position_sha256": file_sha256(position_path),
        "orientation_sha256": file_sha256(orientation_path),
    }
    log_path = Path(cfg.log_jsonl) if cfg.log_jsonl else Path(out).with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if start_update == 0:
        log_path.write_text("", encoding="utf-8")

    def training_state() -> dict[str, Any]:
        return {
            "runner": runner.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    try:
        for local_update in range(cfg.updates):
            update = start_update + local_update
            states = []
            log_probs = []
            entropies = []
            values = []
            rewards = []
            terminateds = []
            truncateds = []
            boundary_next_values = []
            rollout_successes = 0

            for _ in range(cfg.rollout_steps):
                state = _observations_to_state(observations, vsn, cfg.mask_source, str(device))
                action, log_prob, entropy, value = model.sample_action(state)
                vector_step = runner.step(action.detach().cpu().numpy().astype(np.float32))
                done = vector_step.terminated | vector_step.truncated
                rollout_successes += sum(
                    int(is_done and bool(info.get("success")))
                    for is_done, info in zip(done, vector_step.infos, strict=True)
                )
                boundary_values = torch.full((cfg.num_envs,), torch.nan, dtype=value.dtype, device=device)
                boundary_indexes = [
                    index for index, obs in enumerate(vector_step.boundary_observations) if obs is not None
                ]
                if boundary_indexes:
                    with torch.no_grad():
                        boundary_states = _observations_to_state(
                            [vector_step.boundary_observations[index] for index in boundary_indexes],
                            vsn,
                            cfg.mask_source,
                            str(device),
                        )
                        boundary_values[boundary_indexes] = model(boundary_states)[1]

                states.append(state.detach())
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
                rewards.append(torch.as_tensor(vector_step.rewards, dtype=value.dtype, device=device))
                terminateds.append(torch.as_tensor(vector_step.terminated, dtype=torch.bool, device=device))
                truncateds.append(torch.as_tensor(vector_step.truncated, dtype=torch.bool, device=device))
                boundary_next_values.append(boundary_values)
                observations = vector_step.observations
                global_step += cfg.num_envs

            with torch.no_grad():
                next_value = model(_observations_to_state(observations, vsn, cfg.mask_source, str(device)))[1]
                values_t = torch.stack(values)
                rewards_t = torch.stack(rewards)
                terminateds_t = torch.stack(terminateds)
                truncateds_t = torch.stack(truncateds)
                boundary_t = torch.stack(boundary_next_values)
                advantages_by_env = []
                for env_index in range(cfg.num_envs):
                    next_values = []
                    for step_index in range(cfg.rollout_steps):
                        boundary = boundary_t[step_index, env_index]
                        next_values.append(
                            boundary
                            if torch.isfinite(boundary)
                            else values_t[step_index + 1, env_index]
                            if step_index + 1 < cfg.rollout_steps
                            else next_value[env_index]
                        )
                    advantages_by_env.append(
                        compute_gae(
                            rewards_t[:, env_index].tolist(),
                            list(values_t[:, env_index]),
                            next_values,
                            terminateds_t[:, env_index].tolist(),
                            truncateds_t[:, env_index].tolist(),
                            cfg.gamma,
                            cfg.gae_lambda,
                        )
                    )
                advantages_t = torch.stack(advantages_by_env, dim=1)
            returns_t = advantages_t + values_t
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            advantages = returns_t - values_t
            policy_advantages = advantages.detach()
            if policy_advantages.numel() > 1:
                policy_advantages = (policy_advantages - policy_advantages.mean()) / (
                    policy_advantages.std(unbiased=False) + 1e-8
                )

            actor_loss = -(log_probs_t * policy_advantages).mean()
            critic_loss = advantages.pow(2).mean()
            entropy = entropies_t.mean()
            anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
            if anchor_model is not None:
                states_t = torch.stack(states).reshape(-1, 452).to(device)
                current_mean, _current_value = model(states_t)
                with torch.no_grad():
                    anchor_mean, _anchor_value = anchor_model(states_t)
                anchor_loss = torch.nn.functional.mse_loss(current_mean, anchor_mean)
            loss = (
                actor_loss
                + cfg.value_coef * critic_loss
                - cfg.entropy_coef * entropy
                + cfg.anchor_imitation_coef * anchor_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite SFMS loss at update {update}: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            optimizer.step()
            with torch.no_grad():
                model.log_std.clamp_(cfg.min_log_std, cfg.max_log_std)
            metrics = {
                "update": update + 1,
                "global_step": global_step,
                "loss": float(loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "mean_rollout_reward": float(rewards_t.mean().detach().cpu()),
                "rollout_successes": int(rollout_successes),
                "log_std_mean": float(model.log_std.detach().mean().cpu()),
            }
            if cfg.eval_every > 0 and ((update + 1) % cfg.eval_every == 0 or local_update + 1 == cfg.updates):
                eval_result = _deterministic_eval_model(
                    model,
                    vsn,
                    cfg.mask_source,
                    str(device),
                    eval_shapes or shapes or ["synthetic-square"],
                    env_config,
                    camera_config,
                    seed=cfg.eval_seed if cfg.eval_seed is not None else cfg.seed + 100_000,
                    episodes_per_shape=cfg.eval_episodes_per_shape,
                )
                metrics["eval"] = eval_result
                is_better = best_metrics is None or (
                    eval_result["success_rate"],
                    -eval_result["mean_final_xy_error_mm"],
                    -eval_result["mean_final_yaw_error_deg"],
                ) > (
                    best_metrics["success_rate"],
                    -best_metrics["mean_final_xy_error_mm"],
                    -best_metrics["mean_final_yaw_error_deg"],
                )
                if is_better:
                    best_metrics = dict(eval_result)
                    metrics["best_eval"] = best_metrics
                    if best_out is not None:
                        best_checkpoint = make_checkpoint(
                            "SFMSActorCritic",
                            {"input_dim": 452, "hidden1": 256, "hidden2": 128, "action_dim": 3},
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch=update + 1,
                            global_step=global_step,
                            metrics=metrics,
                            data_split={"shapes": shapes or ["synthetic-square"], "mask_source": cfg.mask_source},
                            run=run,
                            train_config=asdict(cfg),
                            compatibility=compatibility,
                            training_state=training_state(),
                        )
                        save_checkpoint(best_out, best_checkpoint)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, default=str) + "\n")
            if cfg.checkpoint_every > 0 and (update + 1) % cfg.checkpoint_every == 0:
                periodic_path = Path(out).with_name(f"{Path(out).stem}.update{update + 1:06d}{Path(out).suffix}")
                periodic = make_checkpoint(
                    "SFMSActorCritic",
                    {"input_dim": 452, "hidden1": 256, "hidden2": 128, "action_dim": 3},
                    model.state_dict(),
                    optimizer.state_dict(),
                    epoch=update + 1,
                    global_step=global_step,
                    metrics=metrics,
                    data_split={"shapes": train_shapes, "mask_source": cfg.mask_source},
                    run=run,
                    train_config=asdict(cfg),
                    compatibility=compatibility,
                    training_state=training_state(),
                )
                save_checkpoint(periodic_path, periodic)
    finally:
        runner.close()

    checkpoint = make_checkpoint(
        "SFMSActorCritic",
        {"input_dim": 452, "hidden1": 256, "hidden2": 128, "action_dim": 3},
        model.state_dict(),
        optimizer.state_dict(),
        epoch=start_update + cfg.updates,
        global_step=global_step,
        metrics=metrics,
        data_split={"shapes": shapes or ["synthetic-square"], "mask_source": cfg.mask_source},
        run=run,
        train_config=asdict(cfg),
        compatibility=compatibility,
        training_state=training_state(),
    )
    save_checkpoint(out, checkpoint)
    metrics["checkpoint"] = str(out)
    if best_out is not None:
        metrics["best_checkpoint"] = str(best_out)
    return metrics


def pretrain_sfms_from_sfss_teacher(
    out: str | Path,
    config: SFMSTeacherPretrainConfig | None = None,
    shapes: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
    vsn: VirtualSensorNetwork | None = None,
) -> dict:
    """Warm-start SFMS by imitating the deterministic SFSS controller.

    This is not the final RL algorithm; it is a practical policy initialization
    so the 452-value SFMS state is mapped to useful continuous actions before
    A2C fine-tuning.  The policy input remains VSN probabilities only.
    """
    torch, _ = _require_torch()
    cfg = config or SFMSTeacherPretrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if cfg.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")

    env = PegInHoleAlignmentEnv(
        shapes=shapes or ["synthetic-square"], seed=cfg.seed, env_config=env_config, camera_config=camera_config
    )
    if vsn is None:
        vsn = VirtualSensorNetwork.from_checkpoints(
            segmentation_path if cfg.mask_source == "predicted" else None,
            position_path,
            orientation_path,
        )
    vsn.to(device).eval()
    teacher = SFSSController(
        max_xy_mm=env.config.max_action_xy_mm,
        max_yaw_deg=env.config.max_action_yaw_deg,
        confidence_mode=cfg.confidence_mode,
    )

    states = []
    targets = []
    try:
        for i in range(cfg.samples):
            shape = env.shapes[i % len(env.shapes)]
            obs, _info = env.reset(seed=cfg.seed + i, options={"shape": shape})
            with torch.no_grad():
                if cfg.mask_source == "ground_truth":
                    mask = torch.as_tensor(obs["mask"][None], dtype=torch.long, device=device)
                    out_vsn = vsn(mask=mask)
                elif cfg.mask_source == "predicted":
                    rgb = torch.as_tensor(obs["rgb"][None], dtype=torch.float32, device=device)
                    out_vsn = vsn(rgb=rgb)
                else:
                    raise ValueError("mask_source must be ground_truth or predicted")
                states.append(make_sfms_state(out_vsn).squeeze(0).cpu())
                targets.append(torch.as_tensor(teacher.act(out_vsn).normalized, dtype=torch.float32))
    finally:
        env.close()

    x = torch.stack(states).to(device)
    y = torch.stack(targets).to(device)
    model = SFMSActorCritic().to(device)
    with torch.no_grad():
        # The teacher-pretrained actor is already useful; keep subsequent RL
        # exploration local instead of sampling almost-uniform clipped actions
        # from std=1.0.
        model.log_std.fill_(-2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = x.shape[0]
    last_loss = 0.0
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            pred, _value = model(x[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0

    metrics = {
        "mode": "sfss_teacher_pretrain",
        "samples": int(cfg.samples),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "loss": last_loss,
        "mask_source": cfg.mask_source,
    }
    checkpoint = make_checkpoint(
        "SFMSActorCritic",
        {"input_dim": 452, "hidden1": 256, "hidden2": 128, "action_dim": 3},
        model.state_dict(),
        optimizer.state_dict(),
        epoch=cfg.epochs,
        global_step=cfg.samples * cfg.epochs,
        metrics=metrics,
        data_split={
            "shapes": shapes or ["synthetic-square"],
            "mask_source": cfg.mask_source,
            "teacher": "SFSSController",
        },
    )
    save_checkpoint(out, checkpoint)
    metrics["checkpoint"] = str(out)
    return metrics
