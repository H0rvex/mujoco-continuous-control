from __future__ import annotations

import json

import gymnasium as gym
import torch

from mujoco_continuous_control.checkpointing import save_checkpoint
from mujoco_continuous_control.evaluate import evaluate_checkpoint
from mujoco_continuous_control.models import ActorCritic
from mujoco_continuous_control.normalization import RunningMeanStd


def _make_pendulum_model(hidden_sizes: tuple[int, ...] = (16,)) -> ActorCritic:
    env = gym.make("Pendulum-v1")
    try:
        return ActorCritic(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
            action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
            hidden_sizes=hidden_sizes,
        )
    finally:
        env.close()


def test_evaluate_checkpoint_writes_json_summary(tmp_path) -> None:
    model = _make_pendulum_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs_rms = RunningMeanStd(shape=(3,))
    obs_rms.update(torch.randn(32, 3))
    checkpoint_path = (
        tmp_path / "runs" / "Pendulum-v1" / "eval" / "checkpoints" / "best.pt"
    )
    output_path = tmp_path / "eval_results.json"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        global_step=128,
        config={
            "env_id": "Pendulum-v1",
            "seed": 1,
            "hidden_sizes": [16],
            "activation": "tanh",
            "log_std_init": -0.5,
            "normalize_obs": True,
        },
        obs_rms=obs_rms,
        extra={"best_eval_return": -10.5},
    )

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        episodes=1,
        seed=1000,
        output=output_path,
        device_name="cpu",
    )

    assert output_path.exists()
    assert results["env_id"] == "Pendulum-v1"
    assert results["deterministic"] is True
    assert results["obs_normalization_loaded"] is True
    assert results["global_step"] == 128
    assert results["best_eval_return"] == -10.5
    assert len(results["episode_returns"]) == 1
    assert len(results["episode_lengths"]) == 1
    assert set(results["summary"]) == {
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
    }

    with output_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["summary"] == results["summary"]
    assert saved["episode_returns"] == results["episode_returns"]
    assert saved["episode_lengths"] == results["episode_lengths"]


def test_evaluate_checkpoint_default_output_path(tmp_path) -> None:
    model = _make_pendulum_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    checkpoint_path = tmp_path / "Pendulum-v1" / "run" / "checkpoints" / "latest.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        global_step=8,
        config={
            "env_id": "Pendulum-v1",
            "seed": 1,
            "hidden_sizes": [16],
            "activation": "tanh",
            "log_std_init": -0.5,
        },
    )

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        episodes=1,
        seed=1000,
        device_name="cpu",
    )

    expected_output = tmp_path / "Pendulum-v1" / "run" / "eval_results.json"
    assert results["output_path"] == str(expected_output)
    assert expected_output.exists()
