from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from mujoco_continuous_control.checkpointing import load_checkpoint
from mujoco_continuous_control.evaluate import (
    _checkpoint_config,
    _env_id,
    _load_checkpoint_payload,
    _load_obs_rms,
    _make_model_from_env,
    _normalize_obs,
    _resolve_device,
)


def _frame_to_image(frame: Any) -> Image.Image:
    frame_array = np.asarray(frame)
    if frame_array.dtype != np.uint8:
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_array)


def _save_gif(frames: list[Any], path: Path, fps: int) -> None:
    if not frames:
        msg = "Cannot save video without rendered frames."
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / fps))
    images = [_frame_to_image(frame) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _make_render_env(env_id: str, seed: int) -> gym.Env[Any, Any]:
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def record_videos(
    checkpoint_path: str | Path,
    episodes: int,
    output_dir: str | Path = "assets/videos",
    seed: int = 1000,
    device_name: str | None = "auto",
    fps: int = 30,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if episodes < 1:
        msg = f"episodes must be >= 1, got {episodes}"
        raise ValueError(msg)
    if fps < 1:
        msg = f"fps must be >= 1, got {fps}"
        raise ValueError(msg)

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    device = _resolve_device(device_name)
    checkpoint = _load_checkpoint_payload(checkpoint_path, map_location=device)
    config = _checkpoint_config(checkpoint)
    env_id = _env_id(checkpoint, config)
    obs_clip = float(config.get("obs_clip", 10.0))

    probe_env = _make_render_env(env_id, seed)
    try:
        model = _make_model_from_env(probe_env, config=config, device=device)
    finally:
        probe_env.close()

    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        map_location=device,
    )
    obs_rms = _load_obs_rms(checkpoint)
    model.eval()

    video_paths: list[str] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    with torch.no_grad():
        for episode_idx in range(episodes):
            episode_seed = seed + episode_idx
            env = _make_render_env(env_id, episode_seed)
            try:
                obs, _ = env.reset(seed=episode_seed)
                frames = [env.render()]
                done = False
                episode_return = 0.0
                episode_length = 0

                while not done and (max_steps is None or episode_length < max_steps):
                    obs_tensor = _normalize_obs(
                        np.asarray(obs)[None, :],
                        device=device,
                        obs_rms=obs_rms,
                        clip=obs_clip,
                    )
                    action, _, _, _, _, _ = model.get_action_and_value(
                        obs_tensor,
                        deterministic=True,
                    )
                    obs, reward, terminated, truncated, _ = env.step(
                        action.squeeze(0).cpu().numpy()
                    )
                    frames.append(env.render())
                    episode_return += float(reward)
                    episode_length += 1
                    done = bool(terminated or truncated)

                video_path = output_dir / f"{env_id}_episode_{episode_idx + 1}.gif"
                _save_gif(frames, video_path, fps=fps)
                video_paths.append(str(video_path))
                episode_returns.append(episode_return)
                episode_lengths.append(episode_length)
            finally:
                env.close()

    return {
        "checkpoint": str(checkpoint_path),
        "env_id": env_id,
        "seed": int(seed),
        "episodes": int(episodes),
        "deterministic": True,
        "obs_normalization_loaded": obs_rms is not None,
        "output_dir": str(output_dir),
        "video_paths": video_paths,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record deterministic rollout GIFs from a PPO checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint .pt file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of rollout videos to record.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/videos"),
        help="Directory for recorded GIFs.",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Recording seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, etc.",
    )
    parser.add_argument("--fps", type=int, default=30, help="GIF frames per second.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode for quick smoke recordings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = record_videos(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        output_dir=args.output_dir,
        seed=args.seed,
        device_name=args.device,
        fps=args.fps,
        max_steps=args.max_steps,
    )
    print(
        "Video recording completed: "
        f"{len(result['video_paths'])} file(s) under {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
