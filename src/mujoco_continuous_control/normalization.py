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
