from __future__ import annotations

import torch

from mujoco_continuous_control.gae import compute_gae


def test_compute_gae_shapes() -> None:
    rewards = torch.ones(4, 3)
    dones = torch.zeros(4, 3)
    values = torch.zeros(4, 3)
    next_value = torch.zeros(3)
    next_done = torch.zeros(3)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        next_done=next_done,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (4, 3)
    assert returns.shape == (4, 3)


def test_compute_gae_single_step_hand_check() -> None:
    rewards = torch.tensor([[1.0, 2.0]])
    dones = torch.tensor([[0.0, 0.0]])
    values = torch.tensor([[0.25, -0.5]])
    next_value = torch.tensor([0.75, 1.5])
    next_done = torch.tensor([0.0, 0.0])
    gamma = 0.9

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        next_done=next_done,
        gamma=gamma,
        gae_lambda=0.95,
    )

    expected_advantages = rewards[0] + gamma * next_value - values[0]
    expected_returns = expected_advantages + values[0]
    assert torch.allclose(advantages[0], expected_advantages)
    assert torch.allclose(returns[0], expected_returns)


def test_compute_gae_terminal_state_handling() -> None:
    rewards = torch.tensor([[1.0], [1.0]])
    dones = torch.tensor([[0.0], [1.0]])
    values = torch.tensor([[0.5], [10.0]])
    next_value = torch.tensor([100.0])
    next_done = torch.tensor([1.0])

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        next_done=next_done,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_last_advantage = torch.tensor([-9.0])
    expected_first_advantage = torch.tensor([0.5])
    assert torch.allclose(advantages[1], expected_last_advantage)
    assert torch.allclose(advantages[0], expected_first_advantage)
    assert torch.allclose(returns[1], torch.tensor([1.0]))
    assert torch.allclose(returns[0], torch.tensor([1.0]))


def test_compute_gae_no_nans() -> None:
    rewards = torch.randn(8, 4)
    dones = torch.zeros(8, 4)
    dones[3, 1] = 1.0
    dones[6, 2] = 1.0
    values = torch.randn(8, 4)
    next_value = torch.randn(4)
    next_done = torch.zeros(4)

    advantages, returns = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        next_value=next_value,
        next_done=next_done,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()
