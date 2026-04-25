from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor


class RolloutBatch(NamedTuple):
    obs: Tensor
    actions: Tensor
    raw_actions: Tensor
    logprobs: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass
class RolloutBuffer:
    rollout_steps: int
    num_envs: int
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.rollout_steps < 1:
            msg = f"rollout_steps must be >= 1, got {self.rollout_steps}"
            raise ValueError(msg)
        if self.num_envs < 1:
            msg = f"num_envs must be >= 1, got {self.num_envs}"
            raise ValueError(msg)

        self.device = torch.device(self.device)
        rollout_shape = (self.rollout_steps, self.num_envs)
        self.obs = torch.zeros(
            (*rollout_shape, *self.obs_shape),
            dtype=self.dtype,
            device=self.device,
        )
        self.actions = torch.zeros(
            (*rollout_shape, *self.action_shape),
            dtype=self.dtype,
            device=self.device,
        )
        self.raw_actions = torch.zeros_like(self.actions)
        self.logprobs = torch.zeros(rollout_shape, dtype=self.dtype, device=self.device)
        self.rewards = torch.zeros(rollout_shape, dtype=self.dtype, device=self.device)
        self.dones = torch.zeros(rollout_shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(rollout_shape, dtype=self.dtype, device=self.device)
        self.advantages = torch.zeros(
            rollout_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.returns = torch.zeros(rollout_shape, dtype=self.dtype, device=self.device)
        self.step = 0

    def add(
        self,
        obs: Tensor,
        actions: Tensor,
        raw_actions: Tensor,
        logprobs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
    ) -> None:
        if self.step >= self.rollout_steps:
            msg = "RolloutBuffer is full; call reset() before adding more data."
            raise IndexError(msg)

        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.raw_actions[self.step].copy_(raw_actions)
        self.logprobs[self.step].copy_(logprobs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.step += 1

    def set_advantages_and_returns(
        self,
        advantages: Tensor,
        returns: Tensor,
    ) -> None:
        self.advantages.copy_(advantages)
        self.returns.copy_(returns)

    def reset(self) -> None:
        self.step = 0

    def flatten(self) -> RolloutBatch:
        batch_size = self.rollout_steps * self.num_envs
        return RolloutBatch(
            obs=self.obs.reshape(batch_size, *self.obs_shape),
            actions=self.actions.reshape(batch_size, *self.action_shape),
            raw_actions=self.raw_actions.reshape(batch_size, *self.action_shape),
            logprobs=self.logprobs.reshape(batch_size),
            rewards=self.rewards.reshape(batch_size),
            dones=self.dones.reshape(batch_size),
            values=self.values.reshape(batch_size),
            advantages=self.advantages.reshape(batch_size),
            returns=self.returns.reshape(batch_size),
        )
