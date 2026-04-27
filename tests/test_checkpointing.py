from __future__ import annotations

import torch

from mujoco_continuous_control.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from mujoco_continuous_control.models import ActorCritic
from mujoco_continuous_control.normalization import RewardNormalizer, RunningMeanStd


def _make_model() -> ActorCritic:
    return ActorCritic(
        obs_dim=4,
        action_dim=2,
        action_low=torch.tensor([-1.0, -2.0]),
        action_high=torch.tensor([1.0, 2.0]),
        hidden_sizes=(16,),
    )


def _train_one_step(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    obs = torch.randn(8, 4)
    _, log_prob, entropy, value, _, _ = model.get_action_and_value(obs)
    loss = -(log_prob + 0.01 * entropy).mean() + value.pow(2).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_save_checkpoint_creates_file(tmp_path) -> None:
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    path = tmp_path / "checkpoints" / "latest.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        global_step=128,
        config={"env_id": "Pendulum-v1", "seed": 7},
    )

    assert path.exists()
    assert path.is_file()


def test_load_checkpoint_restores_model_params(tmp_path) -> None:
    torch.manual_seed(0)
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    _train_one_step(model, optimizer)
    expected_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, optimizer, 256, {"env_id": "Pendulum-v1", "seed": 0})

    loaded_model = _make_model()
    load_checkpoint(path, loaded_model)

    for key, expected_value in expected_state.items():
        assert torch.equal(loaded_model.state_dict()[key], expected_value)


def test_load_checkpoint_restores_optimizer_state(tmp_path) -> None:
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    _train_one_step(model, optimizer)
    path = tmp_path / "optimizer.pt"
    save_checkpoint(path, model, optimizer, 512, {"env_id": "Pendulum-v1", "seed": 0})

    loaded_model = _make_model()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=1e-5)
    checkpoint = load_checkpoint(path, loaded_model, loaded_optimizer)

    assert checkpoint["optimizer_state_dict"]["state"]
    assert loaded_optimizer.state_dict()["state"]
    assert loaded_optimizer.param_groups[0]["lr"] == 3e-4


def test_checkpoint_metadata_exists(tmp_path) -> None:
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = {
        "env_id": "Walker2d-v5",
        "seed": 11,
        "num_envs": 4,
    }
    path = tmp_path / "metadata.pt"

    saved = save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        global_step=1024,
        config=config,
        extra={"best_eval_return": 123.5},
    )
    checkpoint = load_checkpoint(path, _make_model())

    assert saved["global_step"] == 1024
    assert checkpoint["global_step"] == 1024
    assert checkpoint["config"] == config
    assert checkpoint["env_id"] == "Walker2d-v5"
    assert checkpoint["seed"] == 11
    assert checkpoint["best_eval_return"] == 123.5


def test_checkpoint_preserves_obs_normalization_stats(tmp_path) -> None:
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs_rms = RunningMeanStd(shape=(4,))
    obs_rms.update(torch.randn(64, 4) * 3.0 + 2.0)
    path = tmp_path / "normalization.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        global_step=2048,
        config={"env_id": "Pendulum-v1", "seed": 3},
        obs_rms=obs_rms,
    )
    checkpoint = load_checkpoint(path, _make_model())
    loaded_obs_rms = RunningMeanStd.from_state_dict(
        checkpoint["obs_rms"],
        frozen=True,
    )

    assert torch.allclose(loaded_obs_rms.mean, obs_rms.mean)
    assert torch.allclose(loaded_obs_rms.var, obs_rms.var)
    assert torch.allclose(loaded_obs_rms.count, obs_rms.count)
    assert not loaded_obs_rms.training


def test_checkpoint_preserves_reward_normalizer_stats(tmp_path) -> None:
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    reward_normalizer = RewardNormalizer(num_envs=2, gamma=0.99, clip=10.0)
    reward_normalizer.update_and_normalize(
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([False, True]),
    )
    path = tmp_path / "reward_normalization.pt"

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        global_step=4096,
        config={"env_id": "Pendulum-v1", "seed": 4},
        reward_normalizer=reward_normalizer,
    )
    checkpoint = load_checkpoint(path, _make_model())
    loaded = RewardNormalizer.from_state_dict(checkpoint["reward_normalizer"])

    assert loaded.num_envs == reward_normalizer.num_envs
    assert loaded.gamma == reward_normalizer.gamma
    assert loaded.clip == reward_normalizer.clip
    assert torch.allclose(loaded.returns, reward_normalizer.returns)
    assert torch.allclose(loaded.return_rms.mean, reward_normalizer.return_rms.mean)
    assert torch.allclose(loaded.return_rms.var, reward_normalizer.return_rms.var)
    assert torch.allclose(loaded.return_rms.count, reward_normalizer.return_rms.count)
