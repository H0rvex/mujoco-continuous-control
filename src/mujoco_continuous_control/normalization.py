from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor


class RunningMeanStd:
    """Running mean/std for observation normalization.

    Statistics are stored in float64 for numerical stability, while normalized
    observations are returned in the input tensor dtype.
    """

    def __init__(self, shape: int | tuple[int, ...], eps: float = 1e-4) -> None:
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.mean = torch.zeros(self.shape, dtype=torch.float64)
        self.var = torch.ones(self.shape, dtype=torch.float64)
        self.count = torch.tensor(float(eps), dtype=torch.float64)
        self.training = True

    @property
    def std(self) -> Tensor:
        return torch.sqrt(self.var)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def update(self, x: Tensor) -> None:
        if not self.training:
            return

        x64 = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        if x64.shape[-len(self.shape) :] != self.shape:
            msg = f"Expected trailing observation shape {self.shape}, got {x64.shape}"
            raise ValueError(msg)

        reduce_dims = tuple(range(x64.ndim - len(self.shape)))
        if not reduce_dims:
            x64 = x64.reshape((1, *self.shape))
            reduce_dims = (0,)

        batch_mean = x64.mean(dim=reduce_dims)
        batch_var = x64.var(dim=reduce_dims, unbiased=False)
        batch_count = torch.tensor(
            x64.numel() // int(torch.tensor(self.shape).prod()),
            dtype=torch.float64,
            device=self.mean.device,
        )
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x: Tensor, clip: float = 10.0) -> Tensor:
        input_tensor = torch.as_tensor(x)
        input_dtype = input_tensor.dtype
        input_device = input_tensor.device
        normalized = (
            input_tensor.to(torch.float64)
            - self.mean.to(input_device)
        ) / torch.sqrt(self.var.to(input_device) + 1e-8)
        normalized = torch.clamp(normalized, -clip, clip)
        return normalized.to(dtype=input_dtype)

    def update_and_normalize(self, x: Tensor, clip: float = 10.0) -> Tensor:
        self.update(x)
        return self.normalize(x, clip=clip)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count.clone(),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        frozen: bool = False,
    ) -> None:
        shape = tuple(state_dict["shape"])
        mean = torch.as_tensor(state_dict["mean"], dtype=torch.float64)
        var = torch.as_tensor(state_dict["var"], dtype=torch.float64)
        count = torch.as_tensor(state_dict["count"], dtype=torch.float64)

        if shape != self.shape:
            msg = f"Cannot load stats with shape {shape} into shape {self.shape}"
            raise ValueError(msg)
        self.mean = mean.clone()
        self.var = var.clone()
        self.count = count.clone()
        self.training = not frozen

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        frozen: bool = False,
    ) -> RunningMeanStd:
        instance = cls(shape=tuple(state_dict["shape"]))
        instance.load_state_dict(state_dict, frozen=frozen)
        return instance

    def _update_from_moments(
        self,
        batch_mean: Tensor,
        batch_var: Tensor,
        batch_count: Tensor,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        mean_a = self.var * self.count
        mean_b = batch_var * batch_count
        correction = delta.pow(2) * self.count * batch_count / total_count
        new_var = (mean_a + mean_b + correction) / total_count

        self.mean = new_mean
        self.var = torch.clamp(new_var, min=1e-12)
        self.count = total_count


class RewardNormalizer:
    """Normalize rewards using running statistics of discounted returns."""

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        clip: float = 10.0,
    ) -> None:
        if num_envs < 1:
            msg = f"num_envs must be >= 1, got {num_envs}"
            raise ValueError(msg)
        if clip <= 0.0:
            msg = f"clip must be > 0, got {clip}"
            raise ValueError(msg)

        self.num_envs = int(num_envs)
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.returns = torch.zeros(self.num_envs, dtype=torch.float64)
        self.return_rms = RunningMeanStd(shape=(1,))

    def update_and_normalize(self, rewards: Tensor, dones: Tensor) -> Tensor:
        reward_tensor = torch.as_tensor(rewards)
        done_tensor = torch.as_tensor(dones, device=reward_tensor.device)
        if reward_tensor.shape != (self.num_envs,):
            msg = (
                f"Expected rewards with shape ({self.num_envs},), "
                f"got {reward_tensor.shape}"
            )
            raise ValueError(msg)
        if done_tensor.shape != (self.num_envs,):
            msg = (
                f"Expected dones with shape ({self.num_envs},), "
                f"got {done_tensor.shape}"
            )
            raise ValueError(msg)

        input_dtype = reward_tensor.dtype
        input_device = reward_tensor.device
        returns = self.returns.to(input_device)
        returns = returns * self.gamma + reward_tensor.to(torch.float64)
        self.returns = returns.detach().cpu()
        self.return_rms.update(returns.reshape(-1, 1).cpu())

        std = torch.sqrt(self.return_rms.var.to(input_device).reshape(()) + 1e-8)
        normalized = reward_tensor.to(torch.float64) / std
        normalized = torch.clamp(normalized, -self.clip, self.clip)

        reset_mask = done_tensor.to(dtype=torch.bool, device=input_device)
        self.returns[reset_mask.cpu()] = 0.0
        return normalized.to(dtype=input_dtype)

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_envs": self.num_envs,
            "gamma": self.gamma,
            "clip": self.clip,
            "returns": self.returns.clone(),
            "return_rms": self.return_rms.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        num_envs = int(state_dict["num_envs"])
        if num_envs != self.num_envs:
            msg = (
                f"Cannot load reward normalizer with {num_envs} envs "
                f"into {self.num_envs} envs"
            )
            raise ValueError(msg)

        self.gamma = float(state_dict["gamma"])
        self.clip = float(state_dict["clip"])
        returns = torch.as_tensor(state_dict["returns"], dtype=torch.float64)
        if returns.shape != (self.num_envs,):
            msg = (
                f"Expected saved returns with shape ({self.num_envs},), "
                f"got {returns.shape}"
            )
            raise ValueError(msg)
        self.returns = returns.clone()
        self.return_rms.load_state_dict(state_dict["return_rms"])

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, Any]) -> RewardNormalizer:
        instance = cls(
            num_envs=int(state_dict["num_envs"]),
            gamma=float(state_dict["gamma"]),
            clip=float(state_dict["clip"]),
        )
        instance.load_state_dict(state_dict)
        return instance
