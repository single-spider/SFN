"""Multi-frame multi-step (MFMS) recurrent controller utilities.

MFMS observes a short history of the same 452-value VSN state used by SFMS.
This first implementation provides the recurrent model and a teacher-imitation
warm start from the already-successful SFSS controller.  Recurrent RL can build
on this checkpoint instead of starting from random sequence behavior.
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
from ..evaluation.evaluate_sfms import load_sfms_policy
from ..models.controllers import SFSSController
from ..models.vsn import VirtualSensorNetwork
from .common import file_sha256, load_checkpoint_cpu, make_checkpoint, run_metadata, save_checkpoint
from .rl_utils import compute_gae, last_valid_indices, tanh_normal_sample, validity_mask
from .train_sfms import _require_torch, make_sfms_state


@dataclass
class MFMSTrainConfig:
    updates: int = 10
    rollout_steps: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    anchor_imitation_coef: float = 1.0
    grad_norm_clip: float = 0.5
    min_log_std: float = -4.0
    max_log_std: float = 0.5
    eval_every: int = 0
    eval_episodes_per_shape: int = 3
    eval_seed: int | None = None
    checkpoint_every: int = 0
    log_jsonl: str | None = None
    history_len: int = 4
    burn_in_steps: int = 0
    seed: int = 1
    mask_source: str = "ground_truth"
    device: str = "cpu"


@dataclass
class MFMSTeacherPretrainConfig:
    samples: int = 4096
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    history_len: int = 4
    seed: int = 1
    mask_source: str = "ground_truth"
    device: str = "cpu"
    confidence_mode: str = "ignore"


class MFMSActorCritic(_require_torch()[1].Module):
    """LSTM actor-critic baseline from the technical spec."""

    def __init__(
        self,
        input_dim: int = 452,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 3,
    ):
        torch, nn = _require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection = nn.Sequential(nn.Linear(input_dim, projection_dim), nn.ReLU())
        self.lstm = nn.LSTM(projection_dim, hidden_dim, batch_first=True)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def forward(self, x, lengths=None, hidden=None, valid_mask=None):
        """Return actor mean and value for sequence ``x``.

        ``x`` shape is ``[B, T, 452]``.  If lengths are provided, the output is
        taken from each sequence's last valid timestep; otherwise the final
        timestep is used.  This keeps padded episode starts deterministic.
        """
        torch, _ = _require_torch()
        z = self.projection(x)
        out, hidden = self.lstm(z, hidden)
        if valid_mask is not None:
            idx = last_valid_indices(torch.as_tensor(valid_mask, device=x.device))
            last = out[torch.arange(out.shape[0], device=x.device), idx]
        elif lengths is None:
            last = out[:, -1]
        else:
            idx = torch.as_tensor(lengths, dtype=torch.long, device=x.device).clamp_min(1) - 1
            last = out[torch.arange(out.shape[0], device=x.device), idx]
        mean = torch.tanh(self.actor_mean(last))
        value = self.critic(last).squeeze(-1)
        return mean, value, hidden

    def sample_action(self, x, lengths=None, hidden=None, valid_mask=None, deterministic: bool = False):
        torch, _ = _require_torch()
        z = self.projection(x)
        out, hidden = self.lstm(z, hidden)
        if valid_mask is not None:
            idx = last_valid_indices(torch.as_tensor(valid_mask, device=x.device))
        elif lengths is not None:
            idx = torch.as_tensor(lengths, dtype=torch.long, device=x.device).clamp_min(1) - 1
        else:
            idx = torch.full((x.shape[0],), x.shape[1] - 1, dtype=torch.long, device=x.device)
        last = out[torch.arange(out.shape[0], device=x.device), idx]
        action, log_prob, entropy = tanh_normal_sample(self.actor_mean(last), self.log_std, deterministic=deterministic)
        return action, log_prob, entropy, self.critic(last).squeeze(-1), hidden

    def distribution(self, x, lengths=None, hidden=None):
        torch, _ = _require_torch()
        mean, value, hidden = self(x, lengths=lengths, hidden=hidden)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), value, hidden


def make_mfms_history_state(history: list[Any], history_len: int, device: str, return_mask: bool = False):
    """Stack/pad recent 452-value states into ``[1, history_len, 452]``."""
    torch, _ = _require_torch()
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    padded = [torch.zeros(452, dtype=torch.float32, device=device) for _ in range(max(0, history_len - len(history)))]
    recent = [h.reshape(452).to(device) for h in history[-history_len:]]
    sequence = torch.stack(padded + recent, dim=0).unsqueeze(0)
    mask = validity_mask([len(recent)], history_len, padding="left", device=device)
    return (sequence, mask) if return_mask else sequence


def recurrent_context_after_boundary(history: list[Any], terminated: bool, truncated: bool) -> list[Any]:
    """Apply the MFMS episode-boundary contract.

    MFMS reconstructs its LSTM state from a finite observation history on every
    action, so clearing that history resets both effective hidden state and
    sequence context for either Gymnasium boundary type.
    """
    return [] if terminated or truncated else history


def _obs_to_state(obs: dict, vsn: VirtualSensorNetwork, mask_source: str, device: str):
    torch, _ = _require_torch()
    with torch.no_grad():
        if mask_source == "ground_truth":
            out = vsn(mask=torch.as_tensor(obs["mask"][None], dtype=torch.long, device=device))
        elif mask_source == "predicted":
            out = vsn(rgb=torch.as_tensor(obs["rgb"][None], dtype=torch.float32, device=device))
        else:
            raise ValueError("mask_source must be ground_truth or predicted")
        return make_sfms_state(out).squeeze(0).detach(), out


def _deterministic_eval_model(
    model: Any,
    history_len: int,
    vsn: VirtualSensorNetwork,
    mask_source: str,
    device: str,
    shapes: list[str],
    env_config: EnvironmentConfig | None,
    camera_config: CameraConfig | None,
    seed: int,
    episodes_per_shape: int,
) -> dict[str, float]:
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
                history: list[Any] = []
                terminated = truncated = False
                while not (terminated or truncated):
                    state, _out_vsn = _obs_to_state(obs, vsn, mask_source, device)
                    history.append(state)
                    seq, valid_mask = make_mfms_history_state(history, history_len, device, return_mask=True)
                    with torch.no_grad():
                        mean, _value, _hidden = model(seq, valid_mask=valid_mask)
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


def train_mfms(
    out: str | Path,
    config: MFMSTrainConfig | None = None,
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
    """Fine-tune MFMS with A2C from a teacher/imitation checkpoint."""
    torch, _ = _require_torch()
    cfg = config or MFMSTrainConfig()
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

    model_cfg = {
        "input_dim": 452,
        "projection_dim": 256,
        "hidden_dim": 256,
        "action_dim": 3,
        "history_len": cfg.history_len if cfg.history_len > 0 else 4,
    }
    model = MFMSActorCritic().to(device)
    if initial_policy_path is not None and resume_path is not None:
        raise ValueError("initial_policy_path and resume_path are mutually exclusive")
    resume_ckpt = None
    policy_source = resume_path or initial_policy_path
    if policy_source is not None:
        ckpt = load_checkpoint_cpu(policy_source)
        if resume_path is not None:
            resume_ckpt = ckpt
        loaded_cfg = ckpt.get("model_config", {})
        model_cfg.update(
            {k: loaded_cfg[k] for k in ("input_dim", "projection_dim", "hidden_dim", "action_dim") if k in loaded_cfg}
        )
        if "history_len" in loaded_cfg and cfg.history_len <= 0:
            model_cfg["history_len"] = int(loaded_cfg["history_len"])
        model = MFMSActorCritic(
            input_dim=model_cfg["input_dim"],
            projection_dim=model_cfg["projection_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            action_dim=model_cfg["action_dim"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    history_len = int(model_cfg["history_len"])

    anchor_model = None
    if initial_policy_path is not None and cfg.anchor_imitation_coef > 0:
        anchor_model = copy.deepcopy(model).to(device).eval()
        for param in anchor_model.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(model.projection.parameters())
                + list(model.lstm.parameters())
                + list(model.actor_mean.parameters())
                + [model.log_std],
                "lr": cfg.actor_lr,
            },
            {"params": model.critic.parameters(), "lr": cfg.critic_lr},
        ]
    )
    if resume_ckpt is not None and resume_ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    obs, _info = env.reset(seed=cfg.seed, options={"shape": env.shapes[0]})
    history: list[Any] = []
    episode_age = 0
    metrics: dict[str, Any] = {}
    best_metrics: dict[str, Any] | None = None
    global_step = int(resume_ckpt.get("global_step", 0)) if resume_ckpt else 0
    start_update = int(resume_ckpt.get("epoch", 0)) if resume_ckpt else 0
    if resume_ckpt and resume_ckpt.get("training_state"):
        saved = resume_ckpt["training_state"]
        env_state = saved["environment"]
        obs, _info = env.reset(
            seed=int(env_state["seed"]),
            options={"shape": env_state["shape"], "pose_error": env_state["pose_error"], "nontrivial": False},
        )
        env.state.step_count = int(env_state["step_count"])
        env._last_E = float(env_state["last_error_value"])
        env.rng.bit_generator.state = env_state["rng_state"]
        obs = {
            key: value.cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()
            for key, value in saved["observation"].items()
        }
        history = [value.to(device) for value in saved["history"]]
        episode_age = int(saved["episode_age"])
        torch.set_rng_state(saved["torch_rng_state"])
        numpy_state = list(saved["numpy_rng_state"])
        if torch.is_tensor(numpy_state[1]):
            numpy_state[1] = numpy_state[1].cpu().numpy()
        np.random.set_state(tuple(numpy_state))
        if torch.cuda.is_available() and saved.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
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
        if env.state is None:
            raise RuntimeError("MFMS environment is not initialized")
        return {
            "environment": {
                "shape": str(env.state.shape),
                "pose_error": np.asarray(env.state.pose_error, dtype=np.float32).copy(),
                "step_count": int(env.state.step_count),
                "seed": int(env._seed),
                "rng_state": env.rng.bit_generator.state,
                "last_error_value": float(env._last_E),
            },
            "observation": {
                key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value for key, value in obs.items()
            },
            "history": [value.detach().cpu() for value in history],
            "episode_age": int(episode_age),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    try:
        for local_update in range(cfg.updates):
            update = start_update + local_update
            seqs = []
            sequence_masks = []
            loss_masks = []
            log_probs = []
            entropies = []
            values = []
            rewards = []
            terminateds = []
            truncateds = []
            boundary_next_values = []
            rollout_successes = 0

            for _ in range(cfg.rollout_steps):
                state, _out_vsn = _obs_to_state(obs, vsn, cfg.mask_source, str(device))
                history.append(state)
                seq, sequence_mask = make_mfms_history_state(history, history_len, str(device), return_mask=True)
                action, log_prob, entropy, value, _hidden = model.sample_action(seq, valid_mask=sequence_mask)
                loss_eligible = episode_age >= cfg.burn_in_steps

                obs, reward, terminated, truncated, info = env.step(action[0].detach().cpu().numpy().astype(np.float32))
                done = bool(terminated or truncated)
                rollout_successes += int(done and bool(info.get("success")))
                boundary_next_value = None
                if done:
                    with torch.no_grad():
                        final_state, _ = _obs_to_state(obs, vsn, cfg.mask_source, str(device))
                        final_seq, final_mask = make_mfms_history_state(
                            history + [final_state], history_len, str(device), return_mask=True
                        )
                        boundary_next_value = model(final_seq, valid_mask=final_mask)[1].squeeze(0)
                    history = recurrent_context_after_boundary(history, bool(terminated), bool(truncated))
                    episode_age = 0
                    obs, _info = env.reset(
                        seed=cfg.seed + global_step + 1,
                        options={"shape": env.shapes[(global_step + 1) % len(env.shapes)]},
                    )

                seqs.append(seq.squeeze(0).detach())
                sequence_masks.append(sequence_mask.squeeze(0).detach())
                loss_masks.append(loss_eligible)
                log_probs.append(log_prob.squeeze(0))
                entropies.append(entropy.squeeze(0))
                values.append(value.squeeze(0))
                rewards.append(float(reward))
                terminateds.append(bool(terminated))
                truncateds.append(bool(truncated))
                boundary_next_values.append(boundary_next_value)
                global_step += 1
                if not done:
                    episode_age += 1

            with torch.no_grad():
                if not history:
                    bootstrap_state, _ = _obs_to_state(obs, vsn, cfg.mask_source, str(device))
                    bootstrap_history = [bootstrap_state]
                else:
                    bootstrap_history = history
                next_seq, next_mask = make_mfms_history_state(
                    bootstrap_history, history_len, str(device), return_mask=True
                )
                next_value = model(next_seq, valid_mask=next_mask)[1].squeeze(0)
                next_values = [
                    boundary_next_values[i]
                    if boundary_next_values[i] is not None
                    else (values[i + 1] if i + 1 < len(values) else next_value)
                    for i in range(len(values))
                ]
                advantages_t = compute_gae(
                    rewards, values, next_values, terminateds, truncateds, cfg.gamma, cfg.gae_lambda
                )
            values_t = torch.stack(values)
            returns_t = advantages_t + values_t
            log_probs_t = torch.stack(log_probs)
            entropies_t = torch.stack(entropies)
            advantages = returns_t - values_t
            policy_advantages = advantages.detach()
            if policy_advantages.numel() > 1:
                policy_advantages = (policy_advantages - policy_advantages.mean()) / (
                    policy_advantages.std(unbiased=False) + 1e-8
                )

            loss_mask_t = torch.as_tensor(loss_masks, dtype=torch.bool, device=device)
            if not bool(loss_mask_t.any()):
                raise ValueError("burn_in_steps masks the entire rollout; reduce burn-in or increase rollout_steps")
            actor_loss = -(log_probs_t[loss_mask_t] * policy_advantages[loss_mask_t]).mean()
            critic_loss = advantages[loss_mask_t].pow(2).mean()
            entropy = entropies_t[loss_mask_t].mean()
            anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
            if anchor_model is not None:
                seqs_t = torch.stack(seqs).to(device)
                masks_t = torch.stack(sequence_masks).to(device)
                current_mean, _current_value, _ = model(seqs_t, valid_mask=masks_t)
                with torch.no_grad():
                    anchor_mean, _anchor_value, _ = anchor_model(seqs_t, valid_mask=masks_t)
                anchor_loss = torch.nn.functional.mse_loss(current_mean[loss_mask_t], anchor_mean[loss_mask_t])
            loss = (
                actor_loss
                + cfg.value_coef * critic_loss
                - cfg.entropy_coef * entropy
                + cfg.anchor_imitation_coef * anchor_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite MFMS loss at update {update}: {loss.item()}")
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
                "mean_rollout_reward": float(np.mean(rewards)),
                "rollout_successes": int(rollout_successes),
                "log_std_mean": float(model.log_std.detach().mean().cpu()),
                "history_len": history_len,
            }
            if cfg.eval_every > 0 and ((update + 1) % cfg.eval_every == 0 or local_update + 1 == cfg.updates):
                eval_result = _deterministic_eval_model(
                    model,
                    history_len,
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
                            "MFMSActorCritic",
                            model_cfg,
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
                    "MFMSActorCritic",
                    model_cfg,
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
                save_checkpoint(periodic_path, periodic)
    finally:
        env.close()

    checkpoint = make_checkpoint(
        "MFMSActorCritic",
        model_cfg,
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


def pretrain_mfms_from_sfss_teacher(
    out: str | Path,
    config: MFMSTeacherPretrainConfig | None = None,
    shapes: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
    vsn: VirtualSensorNetwork | None = None,
) -> dict:
    """Warm-start MFMS by imitating SFSS from padded state histories."""
    torch, _ = _require_torch()
    cfg = config or MFMSTeacherPretrainConfig()
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

    seqs = []
    targets = []
    try:
        episode = 0
        while len(seqs) < cfg.samples:
            shape = env.shapes[episode % len(env.shapes)]
            obs, _info = env.reset(seed=cfg.seed + episode, options={"shape": shape})
            history: list[Any] = []
            terminated = truncated = False
            while not (terminated or truncated) and len(seqs) < cfg.samples:
                state, out_vsn = _obs_to_state(obs, vsn, cfg.mask_source, str(device))
                history.append(state)
                action = teacher.act(out_vsn)
                seqs.append(make_mfms_history_state(history, cfg.history_len, str(device)).squeeze(0).cpu())
                targets.append(torch.as_tensor(action.normalized, dtype=torch.float32))
                obs, _reward, terminated, truncated, _info = env.step(action.normalized)
            episode += 1
    finally:
        env.close()

    x = torch.stack(seqs).to(device)
    y = torch.stack(targets).to(device)
    model = MFMSActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = x.shape[0]
    last_loss = 0.0
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            # Histories are left-padded, so the current observation is always
            # the final timestep.  Do not pass lengths here; lengths would be
            # appropriate for right-padded batches.
            pred, _value, _hidden = model(x[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0

    metrics = {
        "mode": "sfss_teacher_pretrain",
        "samples": int(cfg.samples),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "history_len": int(cfg.history_len),
        "loss": last_loss,
        "mask_source": cfg.mask_source,
    }
    checkpoint = make_checkpoint(
        "MFMSActorCritic",
        {"input_dim": 452, "projection_dim": 256, "hidden_dim": 256, "action_dim": 3, "history_len": cfg.history_len},
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


def pretrain_mfms_from_sfms_teacher(
    out: str | Path,
    sfms_teacher_path: str | Path,
    config: MFMSTeacherPretrainConfig | None = None,
    shapes: list[str] | None = None,
    env_config: EnvironmentConfig | None = None,
    camera_config: CameraConfig | None = None,
    segmentation_path: str | Path | None = None,
    position_path: str | Path | None = None,
    orientation_path: str | Path | None = None,
    vsn: VirtualSensorNetwork | None = None,
    target_vsn: VirtualSensorNetwork | None = None,
) -> dict:
    """Warm-start MFMS by imitating a successful SFMS policy over rollouts.

    ``vsn`` controls the input history seen by MFMS.  ``target_vsn`` optionally
    controls the state used to query the SFMS teacher.  Supplying a clean
    ``target_vsn`` while ``vsn`` is disturbed trains a simple denoising policy:
    act from disturbed history, but copy the action the teacher would have taken
    from the clean perception state.
    """
    torch, _ = _require_torch()
    cfg = config or MFMSTeacherPretrainConfig()
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
    if target_vsn is not None:
        target_vsn.to(device).eval()
    teacher = load_sfms_policy(sfms_teacher_path, str(device)).eval()

    seqs = []
    targets = []
    try:
        episode = 0
        while len(seqs) < cfg.samples:
            shape = env.shapes[episode % len(env.shapes)]
            obs, _info = env.reset(seed=cfg.seed + episode, options={"shape": shape})
            history: list[Any] = []
            terminated = truncated = False
            while not (terminated or truncated) and len(seqs) < cfg.samples:
                state, _out_vsn = _obs_to_state(obs, vsn, cfg.mask_source, str(device))
                history.append(state)
                with torch.no_grad():
                    teacher_state = state.reshape(1, 452)
                    if target_vsn is not None:
                        teacher_state, _ = _obs_to_state(obs, target_vsn, cfg.mask_source, str(device))
                        teacher_state = teacher_state.reshape(1, 452)
                    mean, _value = teacher(teacher_state)
                    action = torch.clamp(mean, -1.0, 1.0)[0]
                seqs.append(make_mfms_history_state(history, cfg.history_len, str(device)).squeeze(0).cpu())
                targets.append(action.detach().cpu())
                obs, _reward, terminated, truncated, _info = env.step(action.detach().cpu().numpy().astype(np.float32))
            episode += 1
    finally:
        env.close()

    x = torch.stack(seqs).to(device)
    y = torch.stack(targets).to(device)
    model = MFMSActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n = x.shape[0]
    last_loss = 0.0
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            pred, _value, _hidden = model(x[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else 0.0

    metrics = {
        "mode": "sfms_teacher_pretrain",
        "samples": int(cfg.samples),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "history_len": int(cfg.history_len),
        "loss": last_loss,
        "mask_source": cfg.mask_source,
        "sfms_teacher": str(sfms_teacher_path),
    }
    checkpoint = make_checkpoint(
        "MFMSActorCritic",
        {"input_dim": 452, "projection_dim": 256, "hidden_dim": 256, "action_dim": 3, "history_len": cfg.history_len},
        model.state_dict(),
        optimizer.state_dict(),
        epoch=cfg.epochs,
        global_step=cfg.samples * cfg.epochs,
        metrics=metrics,
        data_split={
            "shapes": shapes or ["synthetic-square"],
            "mask_source": cfg.mask_source,
            "teacher": "SFMSActorCritic",
        },
    )
    save_checkpoint(out, checkpoint)
    metrics["checkpoint"] = str(out)
    return metrics
