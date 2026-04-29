#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENV_ORDER = ("Humanoid-v5", "Ant-v5", "Walker2d-v5")
DEFAULT_TIME_LIMIT = 1000


@dataclass(frozen=True)
class SeedResult:
    env_id: str
    run_name: str
    checkpoint: str
    seed: int | None
    episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    full_horizon_episodes: int
    min_episode_length: int
    max_episode_length: int
    deterministic: bool
    obs_normalization_loaded: bool


@dataclass(frozen=True)
class EnvSummary:
    env_id: str
    train_seeds: int
    episodes_per_seed: int
    mean_return: float
    std_across_seeds: float
    best_run_name: str
    best_mean_return: float
    min_seed_mean: float
    max_seed_mean: float
    full_horizon_episodes: int
    total_episodes: int
    representative_video: str
    curves_dir: str
    representative_video_exists: bool
    curves_dir_exists: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize deterministic MuJoCo evaluation JSON files into "
            "README-ready Markdown tables."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing runs/{env_id}/{run_name}/eval_results.json.",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets"),
        help="Directory containing curve and rollout media artifacts.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT,
        help="Episode length counted as full horizon.",
    )
    parser.add_argument(
        "--env",
        action="append",
        choices=ENV_ORDER,
        help="Environment to include. May be repeated. Defaults to all.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object."
        raise TypeError(msg)
    return data


def _seed_from_run_name(run_name: str) -> int | None:
    marker = "seed"
    if marker not in run_name:
        return None
    suffix = run_name.rsplit(marker, maxsplit=1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _as_float(summary: dict[str, Any], key: str, path: Path) -> float:
    try:
        return float(summary[key])
    except KeyError as exc:
        msg = f"{path} is missing summary.{key}."
        raise KeyError(msg) from exc


def load_seed_result(path: Path, time_limit: int) -> SeedResult:
    data = _load_json(path)
    summary = data.get("summary")
    if not isinstance(summary, dict):
        msg = f"{path} is missing a summary object."
        raise TypeError(msg)

    episode_lengths_raw = data.get("episode_lengths", [])
    if not isinstance(episode_lengths_raw, list) or not episode_lengths_raw:
        msg = f"{path} must contain a non-empty episode_lengths list."
        raise TypeError(msg)
    episode_lengths = [int(length) for length in episode_lengths_raw]

    env_id = str(data.get("env_id", path.parent.parent.name))
    run_name = path.parent.name
    episodes = int(data.get("episodes", len(episode_lengths)))
    return SeedResult(
        env_id=env_id,
        run_name=run_name,
        checkpoint=str(data.get("checkpoint", "")),
        seed=_seed_from_run_name(run_name),
        episodes=episodes,
        mean_return=_as_float(summary, "mean_return", path),
        std_return=_as_float(summary, "std_return", path),
        min_return=_as_float(summary, "min_return", path),
        max_return=_as_float(summary, "max_return", path),
        full_horizon_episodes=sum(
            1 for length in episode_lengths if length >= time_limit
        ),
        min_episode_length=min(episode_lengths),
        max_episode_length=max(episode_lengths),
        deterministic=bool(data.get("deterministic", False)),
        obs_normalization_loaded=bool(data.get("obs_normalization_loaded", False)),
    )


def find_results(
    runs_dir: Path, env_ids: tuple[str, ...], time_limit: int
) -> dict[str, list[SeedResult]]:
    results: dict[str, list[SeedResult]] = {}
    for env_id in env_ids:
        env_dir = runs_dir / env_id
        env_results = [
            load_seed_result(path, time_limit=time_limit)
            for path in sorted(env_dir.glob("*/eval_results.json"))
        ]
        env_results.sort(key=lambda result: (result.seed is None, result.seed or 0))
        results[env_id] = env_results
    return results


def build_summary(
    env_id: str,
    seed_results: list[SeedResult],
    assets_dir: Path,
) -> EnvSummary:
    if not seed_results:
        msg = f"No eval_results.json files found for {env_id}."
        raise ValueError(msg)

    seed_means = [result.mean_return for result in seed_results]
    best = max(seed_results, key=lambda result: result.mean_return)
    representative_video = (
        assets_dir / "videos" / env_id / best.run_name / f"{env_id}_episode_1.gif"
    )
    curves_dir = assets_dir / "curves" / env_id
    return EnvSummary(
        env_id=env_id,
        train_seeds=len(seed_results),
        episodes_per_seed=seed_results[0].episodes,
        mean_return=statistics.mean(seed_means),
        std_across_seeds=statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
        best_run_name=best.run_name,
        best_mean_return=best.mean_return,
        min_seed_mean=min(seed_means),
        max_seed_mean=max(seed_means),
        full_horizon_episodes=sum(
            result.full_horizon_episodes for result in seed_results
        ),
        total_episodes=sum(result.episodes for result in seed_results),
        representative_video=str(representative_video),
        curves_dir=str(curves_dir),
        representative_video_exists=representative_video.exists(),
        curves_dir_exists=curves_dir.exists(),
    )


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _length_note(result: SeedResult) -> str:
    if result.min_episode_length == result.max_episode_length:
        return (
            f"{result.full_horizon_episodes}/{result.episodes} full horizon, "
            f"{result.min_episode_length} steps"
        )
    return (
        f"{result.full_horizon_episodes}/{result.episodes} full horizon, "
        f"{result.min_episode_length}-{result.max_episode_length} steps"
    )


def print_aggregate_table(summaries: list[EnvSummary]) -> None:
    print("## Aggregate Results")
    print()
    print(
        "| Environment | Train seeds | Eval episodes / seed | Mean return | "
        "Std across seeds | Best seed/run | Best mean | Full-horizon episodes | "
        "Curves | Representative video | Artifacts |"
    )
    print("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |")
    for summary in summaries:
        artifact_status = ", ".join(
            (
                "curves ok" if summary.curves_dir_exists else "curves missing",
                "video ok" if summary.representative_video_exists else "video missing",
            )
        )
        print(
            f"| {summary.env_id} | {summary.train_seeds} | "
            f"{summary.episodes_per_seed} | {_fmt(summary.mean_return)} | "
            f"{_fmt(summary.std_across_seeds)} | `{summary.best_run_name}` | "
            f"{_fmt(summary.best_mean_return)} | "
            f"{summary.full_horizon_episodes}/{summary.total_episodes} | "
            f"`{summary.curves_dir}/` | `{summary.representative_video}` | "
            f"{artifact_status} |"
        )
    print()


def print_seed_tables(
    results_by_env: dict[str, list[SeedResult]], env_ids: tuple[str, ...]
) -> None:
    print("## Seed-Level Results")
    for env_id in env_ids:
        seed_results = results_by_env.get(env_id, [])
        if not seed_results:
            continue
        print()
        print(f"### {env_id}")
        print()
        print(
            "| Seed | Run | Eval mean | Eval std | Eval min | Eval max | "
            "Episode lengths | Deterministic | Obs norm |"
        )
        print("| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for result in seed_results:
            seed = result.seed if result.seed is not None else "n/a"
            print(
                f"| {seed} | `{result.run_name}` | {_fmt(result.mean_return)} | "
                f"{_fmt(result.std_return)} | {_fmt(result.min_return)} | "
                f"{_fmt(result.max_return)} | {_length_note(result)} | "
                f"{result.deterministic} | {result.obs_normalization_loaded} |"
            )
    print()


def main() -> None:
    args = parse_args()
    env_ids = tuple(args.env) if args.env else ENV_ORDER
    results_by_env = find_results(
        runs_dir=args.runs_dir,
        env_ids=env_ids,
        time_limit=args.time_limit,
    )
    summaries = [
        build_summary(env_id, results_by_env[env_id], assets_dir=args.assets_dir)
        for env_id in env_ids
        if results_by_env[env_id]
    ]
    if not summaries:
        raise SystemExit("No evaluation results found.")

    print_aggregate_table(summaries)
    print_seed_tables(results_by_env, env_ids)


if __name__ == "__main__":
    main()
