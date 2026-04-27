from __future__ import annotations

import csv

import numpy as np
import torch

from mujoco_continuous_control.train import (
    METRIC_FIELDS,
    _bootstrap_truncated_rewards,
    run_training,
)


class _ValueModel(torch.nn.Module):
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = obs.shape[0]
        zeros = torch.zeros(batch_size, 1, dtype=obs.dtype, device=obs.device)
        value = obs.squeeze(-1) * 2.0
        return (
            zeros,
            torch.zeros(batch_size, dtype=obs.dtype, device=obs.device),
            torch.zeros(batch_size, dtype=obs.dtype, device=obs.device),
            value,
            zeros,
            zeros,
        )


def test_truncated_rewards_bootstrap_from_final_observation() -> None:
    rewards = torch.tensor([1.0, 2.0])
    infos = {
        "final_obs": np.array(
            [
                np.array([3.0], dtype=np.float32),
                np.array([5.0], dtype=np.float32),
            ],
            dtype=object,
        ),
        "_final_obs": np.array([True, True]),
    }

    adjusted = _bootstrap_truncated_rewards(
        rewards=rewards,
        truncations=np.array([True, False]),
        infos=infos,
        model=_ValueModel(),
        device=torch.device("cpu"),
        obs_rms=None,
        obs_clip=10.0,
        gamma=0.5,
    )

    assert torch.allclose(adjusted, torch.tensor([4.0, 2.0]))


def test_truncated_reward_bootstrap_is_noop_without_final_observation() -> None:
    rewards = torch.tensor([1.0])

    adjusted = _bootstrap_truncated_rewards(
        rewards=rewards,
        truncations=np.array([True]),
        infos={},
        model=_ValueModel(),
        device=torch.device("cpu"),
        obs_rms=None,
        obs_clip=10.0,
        gamma=0.99,
    )

    assert adjusted is rewards


def test_run_training_writes_run_artifacts(tmp_path) -> None:
    summary = run_training(
        {
            "env_id": "Pendulum-v1",
            "seed": 5,
            "run_name": "phase9_smoke",
            "total_timesteps": 8,
            "num_envs": 2,
            "rollout_steps": 4,
            "num_minibatches": 2,
            "update_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 3.0e-4,
            "anneal_lr": False,
            "normalize_obs": True,
            "normalize_advantages": True,
            "clip_value_loss": True,
            "hidden_sizes": [16],
            "activation": "tanh",
            "log_std_init": -0.5,
            "eval_interval": 8,
            "num_eval_episodes": 1,
            "checkpoint_interval": 8,
            "device": "cpu",
        },
        runs_dir=tmp_path,
    )

    run_dir = summary["run_dir"]
    checkpoint_dir = run_dir / "checkpoints"

    assert run_dir == tmp_path / "Pendulum-v1" / "phase9_smoke"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "plots").is_dir()
    assert (run_dir / "videos").is_dir()
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "final.pt").exists()
    assert (checkpoint_dir / "step_8.pt").exists()
    assert summary["best_checkpoint"] == checkpoint_dir / "best.pt"
    assert summary["latest_checkpoint"] == checkpoint_dir / "latest.pt"

    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0].keys() == set(METRIC_FIELDS)
    assert rows[-1]["global_step"] == "8"
    assert rows[-1]["eval/mean_return"] != "nan"


def test_lr_anneal_timesteps_can_outlive_training_horizon(tmp_path) -> None:
    summary = run_training(
        {
            "env_id": "Pendulum-v1",
            "seed": 7,
            "run_name": "anneal_horizon_smoke",
            "total_timesteps": 12,
            "lr_anneal_timesteps": 16,
            "num_envs": 2,
            "rollout_steps": 4,
            "num_minibatches": 2,
            "update_epochs": 1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "learning_rate": 3.0e-4,
            "anneal_lr": True,
            "normalize_obs": True,
            "normalize_advantages": True,
            "clip_value_loss": True,
            "hidden_sizes": [16],
            "activation": "tanh",
            "log_std_init": -0.5,
            "eval_interval": 0,
            "num_eval_episodes": 0,
            "checkpoint_interval": 0,
            "device": "cpu",
        },
        runs_dir=tmp_path,
    )

    with summary["metrics_path"].open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[-1]["global_step"] == "12"
    assert abs(float(rows[-1]["optimizer/learning_rate"]) - 1.5e-4) < 1e-12
