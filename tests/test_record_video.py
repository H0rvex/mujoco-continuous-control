from __future__ import annotations

import gymnasium as gym
import torch

from mujoco_continuous_control.checkpointing import save_checkpoint
from mujoco_continuous_control.models import ActorCritic
from mujoco_continuous_control.normalization import RunningMeanStd
from mujoco_continuous_control.record_video import record_videos


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


def test_record_videos_writes_deterministic_gif(tmp_path) -> None:
    model = _make_pendulum_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    obs_rms = RunningMeanStd(shape=(3,))
    obs_rms.update(torch.randn(16, 3))
    checkpoint_path = tmp_path / "Pendulum-v1" / "run" / "checkpoints" / "best.pt"
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        global_step=32,
        config={
            "env_id": "Pendulum-v1",
            "seed": 1,
            "hidden_sizes": [16],
            "activation": "tanh",
            "log_std_init": -0.5,
            "normalize_obs": True,
        },
        obs_rms=obs_rms,
    )

    result = record_videos(
        checkpoint_path=checkpoint_path,
        episodes=1,
        output_dir=tmp_path / "assets" / "videos",
        seed=1000,
        device_name="cpu",
        fps=10,
        max_steps=2,
    )

    assert result["deterministic"] is True
    assert result["obs_normalization_loaded"] is True
    assert len(result["video_paths"]) == 1
    assert len(result["episode_returns"]) == 1
    assert result["episode_lengths"] == [2]

    video_path = tmp_path / "assets" / "videos" / "Pendulum-v1_episode_1.gif"
    assert result["video_paths"] == [str(video_path)]
    assert video_path.exists()
    assert video_path.stat().st_size > 0
