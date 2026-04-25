from __future__ import annotations

import torch
from torch import Tensor


def compute_gae(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    next_value: Tensor,
    next_done: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute Generalized Advantage Estimation for vectorized rollouts."""

    if rewards.shape != dones.shape or rewards.shape != values.shape:
        msg = (
            "rewards, dones, and values must have matching "
            f"[rollout_steps, num_envs] shapes; got rewards={rewards.shape}, "
            f"dones={dones.shape}, values={values.shape}"
        )
        raise ValueError(msg)
    if rewards.ndim != 2:
        msg = f"rewards must be rank-2 [rollout_steps, num_envs], got {rewards.ndim}"
        raise ValueError(msg)
    if next_value.shape != rewards.shape[1:]:
        msg = (
            f"next_value must have shape ({rewards.shape[1]},), "
            f"got {next_value.shape}"
        )
        raise ValueError(msg)
    if next_done.shape != rewards.shape[1:]:
        msg = f"next_done must have shape ({rewards.shape[1]},), got {next_done.shape}"
        raise ValueError(msg)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    rollout_steps = rewards.shape[0]

    for step in reversed(range(rollout_steps)):
        if step == rollout_steps - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[step + 1]
            next_values = values[step + 1]

        delta = rewards[step] + gamma * next_nonterminal * next_values - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns
