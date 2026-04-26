from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,I001


PLOT_SPECS = {
    "training_return": {
        "title": "Training episodic return",
        "metrics": ["train/episodic_return"],
        "ylabel": "Return",
    },
    "evaluation_return": {
        "title": "Evaluation return",
        "metrics": ["eval/mean_return"],
        "ylabel": "Return",
    },
    "losses": {
        "title": "PPO losses",
        "metrics": ["loss/policy_loss", "loss/value_loss"],
        "ylabel": "Loss",
    },
    "entropy": {
        "title": "Policy entropy",
        "metrics": ["loss/entropy"],
        "ylabel": "Entropy",
    },
    "approx_kl": {
        "title": "Approximate KL",
        "metrics": ["loss/approx_kl"],
        "ylabel": "KL",
    },
    "clip_fraction": {
        "title": "Clip fraction",
        "metrics": ["loss/clip_fraction"],
        "ylabel": "Fraction",
    },
    "action_std": {
        "title": "Action std diagnostics",
        "metrics": ["policy/action_std", "policy/log_std_mean"],
        "ylabel": "Std / log std",
    },
}


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _read_metrics(path: Path) -> list[dict[str, float | None]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    parsed_rows: list[dict[str, float | None]] = []
    for row in rows:
        parsed_rows.append({key: _parse_float(value) for key, value in row.items()})
    return parsed_rows


def _metric_series(
    rows: Iterable[dict[str, float | None]],
    metric: str,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for index, row in enumerate(rows):
        value = row.get(metric)
        if value is None:
            continue
        step = row.get("global_step")
        x_values.append(float(index if step is None else step))
        y_values.append(value)
    return x_values, y_values


def _curves_dir(run_dir: Path, output_dir: Path) -> Path:
    env_id = run_dir.parent.name if run_dir.parent.name else "run"
    return output_dir / env_id / run_dir.name


def _plot_metrics(
    rows: list[dict[str, float | None]],
    path: Path,
    title: str,
    metrics: list[str],
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for metric in metrics:
        x_values, y_values = _metric_series(rows, metric)
        if not y_values:
            continue
        ax.plot(x_values, y_values, label=metric)
        plotted = True

    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted and len(metrics) > 1:
        ax.legend()
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_run(
    run_dir: str | Path,
    output_dir: str | Path = "assets/curves",
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        msg = f"Missing metrics.csv: {metrics_path}"
        raise FileNotFoundError(msg)

    rows = _read_metrics(metrics_path)
    curves_dir = _curves_dir(run_dir, output_dir)
    plot_paths: dict[str, str] = {}
    for name, spec in PLOT_SPECS.items():
        path = curves_dir / f"{name}.png"
        _plot_metrics(
            rows=rows,
            path=path,
            title=str(spec["title"]),
            metrics=list(spec["metrics"]),
            ylabel=str(spec["ylabel"]),
        )
        plot_paths[name] = str(path)

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "output_dir": str(curves_dir),
        "plot_paths": plot_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO run metrics.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/curves"),
        help="Root directory for generated curve PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = plot_run(run_dir=args.run_dir, output_dir=args.output_dir)
    print(
        "Plotting completed: "
        f"{len(result['plot_paths'])} file(s) under {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
