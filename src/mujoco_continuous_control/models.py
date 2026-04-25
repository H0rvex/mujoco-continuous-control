from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from mujoco_continuous_control.distributions import SquashedGaussian


def _activation(name: str) -> type[nn.Module]:
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]
    except KeyError as exc:
        msg = f"Unsupported activation '{name}'. Expected one of {sorted(activations)}."
        raise ValueError(msg) from exc


def _orthogonal_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _build_trunk(
    input_dim: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(_orthogonal_init(nn.Linear(last_dim, hidden_dim), math.sqrt(2.0)))
        layers.append(activation())
        last_dim = hidden_dim
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """PPO actor-critic with a tanh-squashed Gaussian policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: Tensor,
        action_high: Tensor,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "tanh",
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if obs_dim < 1:
            msg = f"obs_dim must be >= 1, got {obs_dim}"
            raise ValueError(msg)
        if action_dim < 1:
            msg = f"action_dim must be >= 1, got {action_dim}"
            raise ValueError(msg)

        activation_cls = _activation(activation)
        hidden_output_dim = hidden_sizes[-1] if hidden_sizes else obs_dim

        self.actor_trunk = _build_trunk(obs_dim, hidden_sizes, activation_cls)
        self.actor_mean = _orthogonal_init(
            nn.Linear(hidden_output_dim, action_dim),
            gain=0.01,
        )
        self.critic_trunk = _build_trunk(obs_dim, hidden_sizes, activation_cls)
        self.critic_value = _orthogonal_init(
            nn.Linear(hidden_output_dim, 1),
            gain=1.0,
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low_tensor.shape != (action_dim,):
            msg = (
                f"action_low must have shape ({action_dim},), "
                f"got {action_low_tensor.shape}"
            )
            raise ValueError(msg)
        if action_high_tensor.shape != (action_dim,):
            msg = (
                f"action_high must have shape ({action_dim},), "
                f"got {action_high_tensor.shape}"
            )
            raise ValueError(msg)
        if not torch.all(action_high_tensor > action_low_tensor):
            msg = "Every action_high element must be greater than action_low."
            raise ValueError(msg)

        self.register_buffer("action_low", action_low_tensor)
        self.register_buffer("action_high", action_high_tensor)

    def get_action_and_value(
        self,
        obs: Tensor,
        action: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return action, policy diagnostics, value, and raw action.

        ``action`` is interpreted as the pre-tanh raw action. This supports PPO
        updates that recompute log-probs from raw actions stored during rollout.
        """

        mean = self.actor_mean(self.actor_trunk(obs))
        log_std = self.log_std.expand_as(mean)
        distribution = SquashedGaussian(
            mean=mean,
            log_std=log_std,
            action_low=self.action_low,
            action_high=self.action_high,
        )

        if action is not None:
            raw_action = action
            squashed_action = torch.tanh(raw_action)
            env_action = distribution.scale_to_action_space(squashed_action)
        elif deterministic:
            raw_action = mean
            squashed_action = torch.tanh(raw_action)
            env_action = distribution.scale_to_action_space(squashed_action)
        else:
            env_action, raw_action, squashed_action = distribution.sample()

        log_prob = distribution.log_prob(raw_action, squashed_action)
        entropy = distribution.entropy()
        value = self.critic_value(self.critic_trunk(obs)).squeeze(-1)
        return env_action, log_prob, entropy, value, raw_action, squashed_action
