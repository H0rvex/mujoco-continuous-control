from __future__ import annotations

import csv

from mujoco_continuous_control.train import METRIC_FIELDS, run_training


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

    with (run_dir / "metrics.csv").open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows
    assert rows[0].keys() == set(METRIC_FIELDS)
    assert rows[-1]["global_step"] == "8"
    assert rows[-1]["eval/mean_return"] != "nan"
