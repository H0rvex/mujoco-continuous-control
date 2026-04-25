from __future__ import annotations

import torch

from mujoco_continuous_control.distributions import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    SquashedGaussian,
)


def _make_distribution() -> SquashedGaussian:
    mean = torch.tensor(
        [
            [0.0, 0.5, -0.5],
            [1.0, -1.0, 0.25],
            [-0.75, 0.25, 1.5],
            [2.0, -2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    log_std = torch.full_like(mean, -0.5)
    action_low = torch.tensor([-2.0, -1.0, -0.25])
    action_high = torch.tensor([1.0, 3.0, 0.75])
    return SquashedGaussian(mean, log_std, action_low, action_high)


def test_sampled_env_actions_have_correct_shape_and_bounds() -> None:
    distribution = _make_distribution()

    env_action, raw_action, squashed_action = distribution.rsample()

    assert env_action.shape == (4, 3)
    assert raw_action.shape == (4, 3)
    assert squashed_action.shape == (4, 3)
    assert torch.all(env_action >= distribution.action_low)
    assert torch.all(env_action <= distribution.action_high)


def test_log_probs_have_batch_shape_and_no_nans() -> None:
    distribution = _make_distribution()
    env_action, raw_action, squashed_action = distribution.rsample()

    log_prob = distribution.log_prob(raw_action, squashed_action)
    entropy = distribution.entropy()

    assert env_action.shape == (4, 3)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_deterministic_mode_returns_bounded_actions() -> None:
    distribution = _make_distribution()

    mode = distribution.mode()

    assert mode.shape == (4, 3)
    assert torch.all(mode >= distribution.action_low)
    assert torch.all(mode <= distribution.action_high)
    assert torch.isfinite(mode).all()


def test_log_std_is_clamped_for_numerical_safety() -> None:
    mean = torch.zeros(2, 2)
    log_std = torch.tensor(
        [
            [LOG_STD_MIN - 10.0, LOG_STD_MAX + 10.0],
            [0.0, -0.5],
        ]
    )
    distribution = SquashedGaussian(
        mean=mean,
        log_std=log_std,
        action_low=torch.tensor([-1.0, -1.0]),
        action_high=torch.tensor([1.0, 1.0]),
    )

    assert torch.all(distribution.log_std >= LOG_STD_MIN)
    assert torch.all(distribution.log_std <= LOG_STD_MAX)
