from __future__ import annotations

import torch
from torch import Tensor
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SquashedGaussian:
    """Tanh-squashed Gaussian distribution scaled to environment action bounds."""

    def __init__(
        self,
        mean: Tensor,
        log_std: Tensor,
        action_low: Tensor,
        action_high: Tensor,
        eps: float = 1e-6,
    ) -> None:
        self.mean = mean
        self.log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        self.std = self.log_std.exp()
        self.normal = Normal(self.mean, self.std)
        self.eps = eps

        self.action_low = torch.as_tensor(
            action_low,
            dtype=mean.dtype,
            device=mean.device,
        )
        self.action_high = torch.as_tensor(
            action_high,
            dtype=mean.dtype,
            device=mean.device,
        )
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)

    def sample(self) -> tuple[Tensor, Tensor, Tensor]:
        raw_action = self.normal.sample()
        squashed_action = torch.tanh(raw_action)
        env_action = self._scale_to_action_space(squashed_action)
        return env_action, raw_action, squashed_action

    def rsample(self) -> tuple[Tensor, Tensor, Tensor]:
        raw_action = self.normal.rsample()
        squashed_action = torch.tanh(raw_action)
        env_action = self._scale_to_action_space(squashed_action)
        return env_action, raw_action, squashed_action

    def log_prob(
        self,
        raw_action: Tensor,
        squashed_action: Tensor | None = None,
    ) -> Tensor:
        if squashed_action is None:
            squashed_action = torch.tanh(raw_action)

        log_prob = self.normal.log_prob(raw_action)
        log_prob -= torch.log(1.0 - squashed_action.pow(2) + self.eps)
        # The affine map from [-1, 1] to env bounds adds a constant log-det
        # for a fixed action space, so PPO probability ratios do not need it.
        return log_prob.sum(dim=-1)

    def entropy(self) -> Tensor:
        """Return pre-squash Gaussian entropy as a stable diagnostic."""

        return self.normal.entropy().sum(dim=-1)

    def mode(self) -> Tensor:
        squashed_action = torch.tanh(self.mean)
        return self._scale_to_action_space(squashed_action)

    def _scale_to_action_space(self, squashed_action: Tensor) -> Tensor:
        return self.action_bias + self.action_scale * squashed_action
