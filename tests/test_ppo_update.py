from __future__ import annotations

import torch

from mujoco_continuous_control.models import ActorCritic
from mujoco_continuous_control.ppo import ppo_update
from mujoco_continuous_control.rollout import RolloutBatch


def _make_model() -> ActorCritic:
    return ActorCritic(
        obs_dim=4,
        action_dim=2,
        action_low=torch.tensor([-1.0, -2.0]),
        action_high=torch.tensor([1.0, 2.0]),
        hidden_sizes=(32, 32),
    )


def _make_batch(model: ActorCritic, batch_size: int = 16) -> RolloutBatch:
    obs = torch.randn(batch_size, 4)
    with torch.no_grad():
        env_action, logprob, _, value, raw_action, _ = model.get_action_and_value(obs)

    advantages = torch.randn(batch_size)
    returns = value + advantages
    return RolloutBatch(
        obs=obs,
        actions=env_action.detach(),
        raw_actions=raw_action.detach(),
        logprobs=logprob.detach(),
        rewards=torch.randn(batch_size),
        dones=torch.zeros(batch_size),
        values=value.detach(),
        advantages=advantages,
        returns=returns.detach(),
    )


def test_ppo_update_dummy_batch_changes_parameters() -> None:
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    batch = _make_batch(model)
    before = [param.detach().clone() for param in model.parameters()]

    diagnostics = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        config={
            "update_epochs": 2,
            "minibatch_size": 8,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
    )

    assert diagnostics["num_updates"] == 4
    assert any(
        not torch.equal(old_param, new_param)
        for old_param, new_param in zip(before, model.parameters(), strict=True)
    )


def test_ppo_update_diagnostics_keys_and_finite_values() -> None:
    torch.manual_seed(1)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _make_batch(model)

    diagnostics = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        config={"update_epochs": 1, "minibatch_size": 16},
    )

    expected_keys = {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "old_approx_kl",
        "clip_fraction",
        "explained_variance",
        "learning_rate",
        "num_updates",
    }
    assert expected_keys <= diagnostics.keys()
    for key in expected_keys - {"num_updates"}:
        assert torch.isfinite(torch.tensor(diagnostics[key]))
    assert diagnostics["num_updates"] == 1


def test_ppo_update_supports_unclipped_value_loss() -> None:
    torch.manual_seed(2)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _make_batch(model, batch_size=8)

    diagnostics = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        config={
            "update_epochs": 1,
            "minibatch_size": 4,
            "clip_vloss": False,
        },
    )

    assert diagnostics["num_updates"] == 2
    assert torch.isfinite(torch.tensor(diagnostics["value_loss"]))


def test_ppo_update_target_kl_can_stop_early() -> None:
    torch.manual_seed(3)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _make_batch(model, batch_size=16)

    diagnostics = ppo_update(
        model=model,
        optimizer=optimizer,
        batch=batch,
        config={
            "update_epochs": 4,
            "minibatch_size": 4,
            "target_kl": 0.0,
        },
    )

    assert 1 <= diagnostics["num_updates"] < 16
