from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from mujoco_continuous_control.envs import make_vector_env


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        msg = f"Config must be a YAML mapping: {path}"
        raise TypeError(msg)
    return config


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """Run random actions through a vector env.

    This is a packaging and environment-factory smoke path, not PPO training.
    """

    env_id = str(config["env_id"])
    seed = int(config.get("seed", 0))
    num_envs = int(config.get("num_envs", 1))
    total_timesteps = int(config.get("total_timesteps", num_envs))
    gamma = float(config.get("gamma", 0.99))
    capture_video = bool(config.get("capture_video", False))
    run_name = config.get("run_name")
    run_name = str(run_name) if run_name is not None else None

    envs = make_vector_env(
        env_id=env_id,
        seed=seed,
        num_envs=num_envs,
        capture_video=capture_video,
        run_name=run_name,
        gamma=gamma,
    )
    steps = max(1, total_timesteps // num_envs)

    try:
        observations, _ = envs.reset(seed=seed)
        for _ in range(steps):
            actions = [
                envs.single_action_space.sample()
                for _ in range(num_envs)
            ]
            observations, rewards, terminations, truncations, infos = envs.step(actions)
            _ = (observations, rewards, terminations, truncations, infos)
    finally:
        envs.close()

    return {
        "env_id": env_id,
        "num_envs": num_envs,
        "total_vector_steps": steps,
        "total_env_steps": steps * num_envs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-action environment smoke."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke(load_config(args.config))
    print(
        "Smoke run completed: "
        f"{summary['env_id']} with {summary['num_envs']} envs, "
        f"{summary['total_env_steps']} environment steps."
    )


if __name__ == "__main__":
    main()
