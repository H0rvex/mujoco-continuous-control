from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from mujoco_continuous_control.checkpointing import load_checkpoint
from mujoco_continuous_control.envs import make_env
from mujoco_continuous_control.models import ActorCritic
from mujoco_continuous_control.normalization import RunningMeanStd


def _load_checkpoint_payload(
    path: Path,
    map_location: str | torch.device,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        msg = f"Checkpoint must contain a dict, got {type(checkpoint).__name__}."
        raise TypeError(msg)
    return checkpoint


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _as_float_tensor(array: Any, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _normalize_obs(
    obs: Any,
    device: torch.device,
    obs_rms: RunningMeanStd | None,
    clip: float,
) -> Tensor:
    obs_tensor = _as_float_tensor(obs, device)
    if obs_rms is None:
        return obs_tensor
    return obs_rms.normalize(obs_tensor, clip=clip)


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        msg = "Checkpoint config must be a mapping."
        raise TypeError(msg)
    return dict(config)


def _env_id(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    env_id = checkpoint.get("env_id", config.get("env_id"))
    if env_id is None:
        msg = "Checkpoint must include env_id or config.env_id."
        raise ValueError(msg)
    return str(env_id)


def _make_model_from_env(
    env: gym.Env[Any, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> ActorCritic:
    if not isinstance(env.observation_space, gym.spaces.Box):
        msg = "Evaluation requires a continuous Box observation space."
        raise TypeError(msg)
    if not isinstance(env.action_space, gym.spaces.Box):
        msg = "Evaluation requires a continuous Box action space."
        raise TypeError(msg)

    obs_shape = tuple(env.observation_space.shape)
    action_shape = tuple(env.action_space.shape)
    if len(obs_shape) != 1 or len(action_shape) != 1:
        msg = (
            "This evaluator expects flat Box observation and action spaces; "
            f"got obs={obs_shape}, action={action_shape}."
        )
        raise ValueError(msg)

    model = ActorCritic(
        obs_dim=obs_shape[0],
        action_dim=action_shape[0],
        action_low=torch.as_tensor(env.action_space.low, dtype=torch.float32),
        action_high=torch.as_tensor(env.action_space.high, dtype=torch.float32),
        hidden_sizes=tuple(config.get("hidden_sizes", (256, 256))),
        activation=str(config.get("activation", "tanh")),
        log_std_init=float(config.get("log_std_init", -0.5)),
    )
    return model.to(device)


def _load_obs_rms(
    checkpoint: Mapping[str, Any],
) -> RunningMeanStd | None:
    obs_rms_state = checkpoint.get("obs_rms")
    if obs_rms_state is None:
        return None
    if not isinstance(obs_rms_state, Mapping):
        msg = "Checkpoint obs_rms must be a mapping when present."
        raise TypeError(msg)
    return RunningMeanStd.from_state_dict(obs_rms_state, frozen=True)


def _summary(returns: list[float]) -> dict[str, float]:
    values = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(values.mean()),
        "std_return": float(values.std()),
        "min_return": float(values.min()),
        "max_return": float(values.max()),
    }


def _default_output_path(checkpoint_path: Path) -> Path:
    run_dir = checkpoint_path.parent.parent
    if checkpoint_path.parent.name == "checkpoints":
        return run_dir / "eval_results.json"
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_eval_results.json")


def _write_json(path: Path, results: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float):
        return None
    return value_float


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    episodes: int,
    seed: int,
    output: str | Path | None = None,
    device_name: str | None = "auto",
) -> dict[str, Any]:
    if episodes < 1:
        msg = f"episodes must be >= 1, got {episodes}"
        raise ValueError(msg)

    checkpoint_path = Path(checkpoint_path)
    device = _resolve_device(device_name)
    checkpoint = _load_checkpoint_payload(checkpoint_path, map_location=device)
    config = _checkpoint_config(checkpoint)
    env_id = _env_id(checkpoint, config)
    obs_clip = float(config.get("obs_clip", 10.0))

    probe_env = make_env(env_id=env_id, seed=seed)()
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

    returns: list[float] = []
    lengths: list[int] = []
    with torch.no_grad():
        for episode_idx in range(episodes):
            env = make_env(
                env_id=env_id,
                seed=seed,
                idx=episode_idx,
            )()
            try:
                obs, _ = env.reset(seed=seed + episode_idx)
                done = False
                episode_return = 0.0
                episode_length = 0
                while not done:
                    obs_tensor = _normalize_obs(
                        np.asarray(obs)[None, :],
                        device=device,
                        obs_rms=obs_rms,
                        clip=obs_clip,
                    )
                    # deterministic=True follows the Phase 10 protocol:
                    # raw_action = mean, action = tanh(raw_action), then scale.
                    action, _, _, _, _, _ = model.get_action_and_value(
                        obs_tensor,
                        deterministic=True,
                    )
                    obs, reward, terminated, truncated, _ = env.step(
                        action.squeeze(0).cpu().numpy()
                    )
                    episode_return += float(reward)
                    episode_length += 1
                    done = bool(terminated or truncated)
                returns.append(episode_return)
                lengths.append(episode_length)
            finally:
                env.close()

    summary = _summary(returns)
    output_path = (
        Path(output) if output is not None else _default_output_path(checkpoint_path)
    )
    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "env_id": env_id,
        "seed": int(seed),
        "episodes": int(episodes),
        "deterministic": True,
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_eval_return": _finite_or_none(checkpoint.get("best_eval_return")),
        "obs_normalization_loaded": obs_rms is not None,
        "summary": summary,
        "episode_returns": returns,
        "episode_lengths": lengths,
    }
    _write_json(output_path, results)
    results["output_path"] = str(output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO checkpoint deterministically."
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
        default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Evaluation seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to eval_results.json in the run directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, etc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        output=args.output,
        device_name=args.device,
    )
    summary = results["summary"]
    print(
        "Evaluation completed: "
        f"mean={summary['mean_return']:.3f}, "
        f"std={summary['std_return']:.3f}, "
        f"min={summary['min_return']:.3f}, "
        f"max={summary['max_return']:.3f}, "
        f"output={results['output_path']}"
    )


if __name__ == "__main__":
    main()
