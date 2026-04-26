from __future__ import annotations

import csv
from pathlib import Path

from mujoco_continuous_control.plotting import PLOT_SPECS, plot_run
from mujoco_continuous_control.train import METRIC_FIELDS


def test_plot_run_writes_required_curves(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "Pendulum-v1" / "plot_smoke"
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "metrics.csv"
    rows = [
        {
            "global_step": 1,
            "update": 1,
            "train/episodic_return": -100.0,
            "train/episodic_length": 10.0,
            "eval/mean_return": "",
            "eval/std_return": "",
            "eval/min_return": "",
            "eval/max_return": "",
            "loss/policy_loss": 1.0,
            "loss/value_loss": 2.0,
            "loss/entropy": 0.5,
            "loss/approx_kl": 0.01,
            "loss/clip_fraction": 0.1,
            "loss/explained_variance": 0.2,
            "policy/action_mean": 0.0,
            "policy/action_std": 0.8,
            "policy/log_std_mean": -0.5,
            "optimizer/learning_rate": 0.0003,
            "system/fps": 100,
        },
        {
            "global_step": 2,
            "update": 2,
            "train/episodic_return": -80.0,
            "train/episodic_length": 10.0,
            "eval/mean_return": -90.0,
            "eval/std_return": 0.0,
            "eval/min_return": -90.0,
            "eval/max_return": -90.0,
            "loss/policy_loss": 0.8,
            "loss/value_loss": 1.5,
            "loss/entropy": 0.4,
            "loss/approx_kl": 0.02,
            "loss/clip_fraction": 0.2,
            "loss/explained_variance": 0.3,
            "policy/action_mean": 0.0,
            "policy/action_std": 0.7,
            "policy/log_std_mean": -0.6,
            "optimizer/learning_rate": 0.0003,
            "system/fps": 100,
        },
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = plot_run(run_dir=run_dir, output_dir=tmp_path / "assets" / "curves")

    assert set(result["plot_paths"]) == set(PLOT_SPECS)
    for plot_path in result["plot_paths"].values():
        path = Path(plot_path)
        assert path.exists()
        assert path.stat().st_size > 0
