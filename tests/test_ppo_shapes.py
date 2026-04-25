from __future__ import annotations

import pytest
import torch

from mujoco_continuous_control.models import ActorCritic


def _make_model() -> ActorCritic:
    return ActorCritic(
        obs_dim=5,
        action_dim=3,
        action_low=torch.tensor([-2.0, -1.0, -0.5]),
        action_high=torch.tensor([1.0, 3.0, 0.5]),
        hidden_sizes=(32, 32),
    )


def test_actor_critic_action_value_shapes_and_bounds() -> None:
    model = _make_model()
    obs = torch.randn(7, 5)

    env_action, log_prob, entropy, value, raw_action, squashed_action = (
        model.get_action_and_value(obs)
    )

    assert env_action.shape == (7, 3)
    assert log_prob.shape == (7,)
    assert entropy.shape == (7,)
    assert value.shape == (7,)
    assert raw_action.shape == (7, 3)
    assert squashed_action.shape == (7, 3)
    assert torch.all(env_action >= model.action_low)
    assert torch.all(env_action <= model.action_high)
    assert torch.isfinite(env_action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()


def test_actor_critic_deterministic_mode_is_bounded() -> None:
    model = _make_model()
    obs = torch.randn(4, 5)

    env_action, log_prob, entropy, value, raw_action, squashed_action = (
        model.get_action_and_value(obs, deterministic=True)
    )

    assert env_action.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4,)
    assert raw_action.shape == (4, 3)
    assert squashed_action.shape == (4, 3)
    assert torch.all(env_action >= model.action_low)
    assert torch.all(env_action <= model.action_high)
    assert torch.isfinite(env_action).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_actor_critic_recomputes_log_prob_from_raw_action() -> None:
    model = _make_model()
    obs = torch.randn(6, 5)
    _, _, _, _, raw_action, _ = model.get_action_and_value(obs)

    env_action, log_prob, entropy, value, returned_raw_action, squashed_action = (
        model.get_action_and_value(obs, action=raw_action)
    )

    assert torch.equal(returned_raw_action, raw_action)
    assert env_action.shape == (6, 3)
    assert log_prob.shape == (6,)
    assert entropy.shape == (6,)
    assert value.shape == (6,)
    assert squashed_action.shape == (6, 3)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(value).all()


def test_actor_critic_global_log_std_shape() -> None:
    model = _make_model()

    assert model.log_std.shape == (3,)
    assert model.log_std.requires_grad


def test_actor_critic_rejects_unknown_activation() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        ActorCritic(
            obs_dim=5,
            action_dim=3,
            action_low=torch.tensor([-1.0, -1.0, -1.0]),
            action_high=torch.tensor([1.0, 1.0, 1.0]),
            activation="gelu",
        )
