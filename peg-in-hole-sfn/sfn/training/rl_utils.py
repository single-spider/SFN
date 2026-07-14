"""Small, testable RL math utilities shared by SFMS and MFMS."""

from __future__ import annotations

import torch


def tanh_normal_sample(mean, log_std, *, deterministic: bool = False, epsilon: float = 1e-6):
    """Sample a bounded action and return its change-of-variables log density.

    ``mean`` is the *pre-tanh* Normal location.  The returned log probability
    is for the bounded action, unlike clipping a Normal sample after computing
    its unbounded density.
    """
    std = log_std.exp().expand_as(mean)
    dist = torch.distributions.Normal(mean, std)
    raw = mean if deterministic else dist.rsample()
    action = torch.tanh(raw)
    log_prob = (dist.log_prob(raw) - torch.log(1.0 - action.square() + epsilon)).sum(dim=-1)
    # There is no simple closed-form entropy after tanh.  This sample estimate
    # is unbiased for -E[log pi(a)] and retains useful gradients.
    entropy = -log_prob
    return action, log_prob, entropy


def compute_gae(rewards, values, next_values, terminated, truncated, gamma: float, gae_lambda: float):
    """Compute GAE, bootstrapping truncations but never true terminations.

    Truncation ends the trace (preventing leakage into a reset episode), while
    its delta still bootstraps from the final observation value.
    """
    if not (len(rewards) == len(values) == len(next_values) == len(terminated) == len(truncated)):
        raise ValueError("GAE inputs must have equal lengths")
    advantages = []
    gae = torch.zeros_like(values[0]) if values else torch.tensor(0.0)
    for reward, value, next_value, term, trunc in zip(
        reversed(rewards),
        reversed(values),
        reversed(next_values),
        reversed(terminated),
        reversed(truncated),
        strict=False,
    ):
        reward_t = torch.as_tensor(reward, dtype=value.dtype, device=value.device)
        bootstrap = 0.0 if term else 1.0
        trace = 0.0 if (term or trunc) else 1.0
        delta = reward_t + gamma * next_value * bootstrap - value
        gae = delta + gamma * gae_lambda * trace * gae
        advantages.append(gae)
    return torch.stack(list(reversed(advantages))) if advantages else torch.empty(0)


def validity_mask(lengths, max_len: int, *, padding: str = "right", device=None):
    """Build a boolean recurrent validity mask for left or right padding."""
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
    if max_len <= 0 or torch.any(lengths < 0) or torch.any(lengths > max_len):
        raise ValueError("lengths must be between zero and max_len")
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    if padding == "right":
        return steps < lengths.unsqueeze(1)
    if padding == "left":
        return steps >= (max_len - lengths).unsqueeze(1)
    raise ValueError("padding must be 'left' or 'right'")


def last_valid_indices(mask):
    """Return each row's last valid index, rejecting all-padding sequences."""
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if mask.ndim != 2 or torch.any(~mask.any(dim=1)):
        raise ValueError("validity mask must be [B,T] with at least one valid item per row")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(dim=1).values
