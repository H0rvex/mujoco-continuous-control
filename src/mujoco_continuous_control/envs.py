from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym

EnvFactory = Callable[[], gym.Env[Any, Any]]


def make_env(
    env_id: str,
    seed: int,
    idx: int = 0,
    capture_video: bool = False,
    run_name: str | None = None,
    gamma: float = 0.99,
) -> EnvFactory:
    """Create a thunk that builds one seeded Gymnasium environment.

    The thunk shape is intentional: it plugs directly into
    ``gym.vector.SyncVectorEnv`` while keeping video recording scoped to the
    specific worker that requested it.
    """

    def thunk() -> gym.Env[Any, Any]:
        render_mode = "rgb_array" if capture_video else None
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video:
            video_folder = Path("videos") / (run_name or env_id)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_folder),
                episode_trigger=lambda episode_id: episode_id == 0,
                disable_logger=True,
            )

        env_seed = seed + idx
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        env.observation_space.seed(env_seed)
        return env

    return thunk


def make_vector_env(
    env_id: str,
    seed: int,
    num_envs: int,
    capture_video: bool = False,
    run_name: str | None = None,
    gamma: float = 0.99,
) -> gym.vector.SyncVectorEnv:
    """Create a SyncVectorEnv for training or smoke execution."""

    if num_envs < 1:
        msg = f"num_envs must be >= 1, got {num_envs}"
        raise ValueError(msg)

    env_fns = [
        make_env(
            env_id=env_id,
            seed=seed,
            idx=idx,
            capture_video=capture_video and idx == 0,
            run_name=run_name,
            gamma=gamma,
        )
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)
