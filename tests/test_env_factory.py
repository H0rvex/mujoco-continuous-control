from __future__ import annotations

import gymnasium as gym
import pytest

from mujoco_continuous_control.envs import make_env, make_vector_env


def _requires_env(env_id: str) -> None:
    try:
        env = gym.make(env_id)
    except Exception as exc:
        pytest.skip(f"{env_id} is unavailable: {exc}")
    else:
        env.close()


def test_pendulum_vector_env_smoke_creation() -> None:
    envs = make_vector_env(env_id="Pendulum-v1", seed=123, num_envs=2)
    try:
        observations, infos = envs.reset(seed=123)
        actions = [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        next_observations, rewards, terminations, truncations, step_infos = envs.step(
            actions
        )

        assert observations.shape[0] == 2
        assert next_observations.shape[0] == 2
        assert rewards.shape == (2,)
        assert terminations.shape == (2,)
        assert truncations.shape == (2,)
        assert isinstance(infos, dict)
        assert isinstance(step_infos, dict)
    finally:
        envs.close()


@pytest.mark.parametrize("env_id", ["Walker2d-v5", "Ant-v5"])
def test_mujoco_action_spaces_are_continuous(env_id: str) -> None:
    _requires_env(env_id)
    env = make_env(env_id=env_id, seed=7)()
    try:
        assert isinstance(env.unwrapped.action_space, gym.spaces.Box)
        assert env.unwrapped.action_space.shape is not None
        assert len(env.unwrapped.action_space.shape) == 1
    finally:
        env.close()
