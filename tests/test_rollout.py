from __future__ import annotations

import pytest
import torch

from mujoco_continuous_control.rollout import RolloutBuffer


def test_rollout_buffer_flattening_shapes() -> None:
    rollout_steps = 4
    num_envs = 3
    obs_shape = (5,)
    action_shape = (2,)
    buffer = RolloutBuffer(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_shape=obs_shape,
        action_shape=action_shape,
    )

    for step in range(rollout_steps):
        buffer.add(
            obs=torch.full((num_envs, *obs_shape), float(step)),
            actions=torch.ones(num_envs, *action_shape),
            raw_actions=torch.zeros(num_envs, *action_shape),
            logprobs=torch.full((num_envs,), -0.5),
            rewards=torch.arange(num_envs, dtype=torch.float32),
            dones=torch.zeros(num_envs),
            values=torch.full((num_envs,), 0.25),
        )

    advantages = torch.randn(rollout_steps, num_envs)
    returns = torch.randn(rollout_steps, num_envs)
    buffer.set_advantages_and_returns(advantages, returns)

    batch = buffer.flatten()
    batch_size = rollout_steps * num_envs

    assert batch.obs.shape == (batch_size, *obs_shape)
    assert batch.actions.shape == (batch_size, *action_shape)
    assert batch.raw_actions.shape == (batch_size, *action_shape)
    assert batch.logprobs.shape == (batch_size,)
    assert batch.rewards.shape == (batch_size,)
    assert batch.dones.shape == (batch_size,)
    assert batch.values.shape == (batch_size,)
    assert batch.advantages.shape == (batch_size,)
    assert batch.returns.shape == (batch_size,)


def test_rollout_buffer_preserves_flatten_order() -> None:
    buffer = RolloutBuffer(
        rollout_steps=2,
        num_envs=3,
        obs_shape=(1,),
        action_shape=(1,),
    )

    buffer.obs.copy_(torch.tensor([[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]]))
    batch = buffer.flatten()

    assert torch.equal(batch.obs[:, 0], torch.arange(6, dtype=torch.float32))


def test_rollout_buffer_rejects_overfill() -> None:
    buffer = RolloutBuffer(
        rollout_steps=1,
        num_envs=1,
        obs_shape=(2,),
        action_shape=(1,),
    )
    payload = {
        "obs": torch.zeros(1, 2),
        "actions": torch.zeros(1, 1),
        "raw_actions": torch.zeros(1, 1),
        "logprobs": torch.zeros(1),
        "rewards": torch.zeros(1),
        "dones": torch.zeros(1),
        "values": torch.zeros(1),
    }

    buffer.add(**payload)
    with pytest.raises(IndexError, match="RolloutBuffer is full"):
        buffer.add(**payload)
