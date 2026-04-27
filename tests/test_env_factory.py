from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.envs.registration import register, registry
from gymnasium.vector import AutoresetMode

from mujoco_continuous_control.envs import make_env, make_vector_env

_ONE_STEP_ENV_ID = "OneStepContinuous-v0"


class _OneStepContinuousEnv(gym.Env):
    observation_space = gym.spaces.Box(-100.0, 100.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        self.step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        return (
            np.array([float(self.step_count)], dtype=np.float32),
            1.0,
            True,
            False,
            {},
        )


def _register_one_step_env() -> None:
    if _ONE_STEP_ENV_ID not in registry:
        register(
            id=_ONE_STEP_ENV_ID,
            entry_point=_OneStepContinuousEnv,
        )


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


def test_vector_env_uses_same_step_autoreset_without_dummy_transition() -> None:
    _register_one_step_env()
    envs = make_vector_env(env_id=_ONE_STEP_ENV_ID, seed=123, num_envs=1)
    try:
        assert envs.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP

        actions = np.zeros((1, 1), dtype=np.float32)
        envs.reset(seed=123)

        observations, rewards, terminations, truncations, infos = envs.step(actions)
        assert rewards.tolist() == [1.0]
        assert terminations.tolist() == [True]
        assert truncations.tolist() == [False]
        assert observations.tolist() == [[0.0]]
        assert "_final_obs" in infos

        observations, rewards, terminations, truncations, _ = envs.step(actions)
        assert rewards.tolist() == [1.0]
        assert terminations.tolist() == [True]
        assert truncations.tolist() == [False]
        assert observations.tolist() == [[0.0]]
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
