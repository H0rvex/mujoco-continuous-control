from __future__ import annotations

import argparse
import csv
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from torch import Tensor

from mujoco_continuous_control.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from mujoco_continuous_control.envs import make_env, make_vector_env
from mujoco_continuous_control.gae import compute_gae
from mujoco_continuous_control.models import ActorCritic
from mujoco_continuous_control.normalization import RunningMeanStd
from mujoco_continuous_control.ppo import ppo_update
from mujoco_continuous_control.rollout import RolloutBuffer

METRIC_FIELDS = [
    "global_step",
    "update",
    "train/episodic_return",
    "train/episodic_length",
    "eval/mean_return",
    "eval/std_return",
    "eval/min_return",
    "eval/max_return",
    "loss/policy_loss",
    "loss/value_loss",
    "loss/entropy",
    "loss/approx_kl",
    "loss/clip_fraction",
    "loss/explained_variance",
    "policy/action_mean",
    "policy/action_std",
    "policy/log_std_mean",
    "optimizer/learning_rate",
    "system/fps",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        msg = f"Config must be a YAML mapping: {path}"
        raise TypeError(msg)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _default_run_name(env_id: str, seed: int) -> str:
    return f"{env_id.lower().replace('/', '_')}_seed{seed}"


def _prepare_run_dir(
    env_id: str,
    run_name: str,
    runs_dir: str | Path = "runs",
) -> Path:
    run_dir = Path(runs_dir) / env_id / run_name
    for child in ("checkpoints", "plots", "videos"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_config(path: Path, config: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True)


def _as_float_tensor(array: Any, device: torch.device) -> Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _normalize_obs(
    obs: Any,
    device: torch.device,
    obs_rms: RunningMeanStd | None,
    clip: float,
    update: bool,
) -> Tensor:
    obs_tensor = _as_float_tensor(obs, device)
    if obs_rms is None:
        return obs_tensor
    if update:
        return obs_rms.update_and_normalize(obs_tensor, clip=clip)
    return obs_rms.normalize(obs_tensor, clip=clip)


def _extract_episode_metrics(infos: Mapping[str, Any]) -> list[tuple[float, float]]:
    if "episode" not in infos:
        return []

    episode = infos["episode"]
    if not isinstance(episode, Mapping):
        return []

    returns = np.asarray(episode.get("r", []), dtype=np.float64).reshape(-1)
    lengths = np.asarray(episode.get("l", []), dtype=np.float64).reshape(-1)
    if returns.size == 0 or lengths.size == 0:
        return []

    mask = np.asarray(infos.get("_episode", np.ones_like(returns, dtype=bool)))
    mask = mask.astype(bool).reshape(-1)
    return [
        (float(episode_return), float(episode_length))
        for episode_return, episode_length, keep in zip(
            returns,
            lengths,
            mask,
            strict=False,
        )
        if keep
    ]


def _make_model(
    obs_dim: int,
    action_dim: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> ActorCritic:
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=torch.as_tensor(action_low, dtype=torch.float32),
        action_high=torch.as_tensor(action_high, dtype=torch.float32),
        hidden_sizes=tuple(config.get("hidden_sizes", (256, 256))),
        activation=str(config.get("activation", "tanh")),
        log_std_init=float(config.get("log_std_init", -0.5)),
    )
    return model.to(device)


def _ppo_config(config: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    num_minibatches = int(config.get("num_minibatches", 1))
    minibatch_size = int(config.get("minibatch_size", batch_size // num_minibatches))
    minibatch_size = max(1, minibatch_size)
    return {
        "clip_coef": float(config.get("clip_coef", 0.2)),
        "ent_coef": float(config.get("ent_coef", 0.0)),
        "vf_coef": float(config.get("vf_coef", 0.5)),
        "max_grad_norm": float(config.get("max_grad_norm", 0.5)),
        "update_epochs": int(config.get("update_epochs", 1)),
        "minibatch_size": minibatch_size,
        "normalize_advantages": bool(config.get("normalize_advantages", True)),
        "clip_vloss": bool(
            config.get("clip_value_loss", config.get("clip_vloss", True))
        ),
        "target_kl": config.get("target_kl"),
    }


def _evaluation_summary(returns: list[float]) -> dict[str, float]:
    if not returns:
        nan = float("nan")
        return {
            "eval/mean_return": nan,
            "eval/std_return": nan,
            "eval/min_return": nan,
            "eval/max_return": nan,
        }

    values = np.asarray(returns, dtype=np.float64)
    return {
        "eval/mean_return": float(values.mean()),
        "eval/std_return": float(values.std()),
        "eval/min_return": float(values.min()),
        "eval/max_return": float(values.max()),
    }


def evaluate_policy(
    model: ActorCritic,
    env_id: str,
    seed: int,
    episodes: int,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    obs_clip: float = 10.0,
) -> dict[str, float]:
    """Run deterministic evaluation without updating observation statistics."""

    if episodes < 1:
        return _evaluation_summary([])

    returns: list[float] = []
    was_training = model.training
    model.eval()

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
            while not done:
                obs_tensor = _normalize_obs(
                    np.asarray(obs)[None, :],
                    device=device,
                    obs_rms=obs_rms,
                    clip=obs_clip,
                    update=False,
                )
                with torch.no_grad():
                    action, _, _, _, _, _ = model.get_action_and_value(
                        obs_tensor,
                        deterministic=True,
                    )
                obs, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                done = bool(terminated or truncated)
                episode_return += float(reward)
            returns.append(episode_return)
        finally:
            env.close()

    if was_training:
        model.train()
    return _evaluation_summary(returns)


def _append_metrics(
    writer: csv.DictWriter,
    metrics_file: Any,
    metrics: Mapping[str, Any],
) -> None:
    row = {field: metrics.get(field, float("nan")) for field in METRIC_FIELDS}
    writer.writerow(row)
    metrics_file.flush()


def _checkpoint_extra(best_eval_return: float) -> dict[str, float]:
    return {"best_eval_return": float(best_eval_return)}


def run_training(
    config: Mapping[str, Any],
    runs_dir: str | Path = "runs",
) -> dict[str, Any]:
    config = dict(config)
    env_id = str(config["env_id"])
    seed = int(config.get("seed", 1))
    run_name = str(config.get("run_name") or _default_run_name(env_id, seed))
    total_timesteps = int(config.get("total_timesteps", 1_000_000))
    num_envs = int(config.get("num_envs", 1))
    rollout_steps = int(config.get("rollout_steps", 128))
    batch_size = rollout_steps * num_envs
    gamma = float(config.get("gamma", 0.99))
    gae_lambda = float(config.get("gae_lambda", 0.95))
    learning_rate = float(config.get("learning_rate", 3e-4))
    anneal_lr = bool(config.get("anneal_lr", False))
    normalize_obs = bool(config.get("normalize_obs", True))
    obs_clip = float(config.get("obs_clip", 10.0))
    eval_interval = int(config.get("eval_interval", 0))
    num_eval_episodes = int(config.get("num_eval_episodes", 0))
    checkpoint_interval = int(config.get("checkpoint_interval", 0))
    resume_path = config.get("resume")
    device = _resolve_device(config.get("device"))

    if rollout_steps < 1:
        msg = f"rollout_steps must be >= 1, got {rollout_steps}"
        raise ValueError(msg)
    if num_envs < 1:
        msg = f"num_envs must be >= 1, got {num_envs}"
        raise ValueError(msg)
    if total_timesteps < 1:
        msg = f"total_timesteps must be >= 1, got {total_timesteps}"
        raise ValueError(msg)

    config["env_id"] = env_id
    config["seed"] = seed
    config["run_name"] = run_name
    config["total_timesteps"] = total_timesteps
    config["num_envs"] = num_envs
    config["rollout_steps"] = rollout_steps
    config["device"] = str(device)

    seed_everything(seed)
    run_dir = _prepare_run_dir(env_id, run_name, runs_dir=runs_dir)
    _write_config(run_dir / "config.yaml", config)

    envs = make_vector_env(
        env_id=env_id,
        seed=seed,
        num_envs=num_envs,
        gamma=gamma,
    )
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.csv"

    try:
        if not isinstance(envs.single_observation_space, gym.spaces.Box):
            msg = "PPO training requires a continuous Box observation space."
            raise TypeError(msg)
        if not isinstance(envs.single_action_space, gym.spaces.Box):
            msg = "PPO training requires a continuous Box action space."
            raise TypeError(msg)

        obs_shape = tuple(envs.single_observation_space.shape)
        action_shape = tuple(envs.single_action_space.shape)
        if len(obs_shape) != 1 or len(action_shape) != 1:
            msg = (
                "This PPO implementation expects flat Box observation and action "
                f"spaces; got obs={obs_shape}, action={action_shape}."
            )
            raise ValueError(msg)

        model = _make_model(
            obs_dim=obs_shape[0],
            action_dim=action_shape[0],
            action_low=envs.single_action_space.low,
            action_high=envs.single_action_space.high,
            config=config,
            device=device,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=float(config.get("adam_eps", 1e-5)),
        )
        obs_rms = RunningMeanStd(obs_shape) if normalize_obs else None
        global_step = 0
        start_update = 0
        best_eval_return = float("-inf")

        if resume_path is not None:
            checkpoint = load_checkpoint(
                Path(resume_path),
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
            global_step = int(checkpoint.get("global_step", 0))
            start_update = global_step // batch_size
            best_eval_return = float(checkpoint.get("best_eval_return", float("-inf")))
            if obs_rms is not None and checkpoint.get("obs_rms") is not None:
                obs_rms.load_state_dict(checkpoint["obs_rms"])

        next_obs, _ = envs.reset(seed=seed)
        next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)
        train_returns: list[float] = []
        train_lengths: list[float] = []
        next_eval_step = (
            ((global_step // eval_interval) + 1) * eval_interval
            if eval_interval > 0
            else None
        )
        next_checkpoint_step = (
            ((global_step // checkpoint_interval) + 1) * checkpoint_interval
            if checkpoint_interval > 0
            else None
        )
        start_time = time.time()

        with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
            writer.writeheader()

            update = start_update
            while global_step < total_timesteps:
                update += 1
                remaining_timesteps = total_timesteps - global_step
                current_rollout_steps = min(
                    rollout_steps,
                    max(1, (remaining_timesteps + num_envs - 1) // num_envs),
                )
                current_batch_size = current_rollout_steps * num_envs
                if anneal_lr:
                    fraction = max(0.0, 1.0 - global_step / total_timesteps)
                    optimizer.param_groups[0]["lr"] = fraction * learning_rate

                buffer = RolloutBuffer(
                    rollout_steps=current_rollout_steps,
                    num_envs=num_envs,
                    obs_shape=obs_shape,
                    action_shape=action_shape,
                    device=device,
                )
                for _ in range(current_rollout_steps):
                    obs_tensor = _normalize_obs(
                        next_obs,
                        device=device,
                        obs_rms=obs_rms,
                        clip=obs_clip,
                        update=True,
                    )
                    with torch.no_grad():
                        action, logprob, _, value, raw_action, _ = (
                            model.get_action_and_value(obs_tensor)
                        )

                    next_obs, reward, termination, truncation, infos = envs.step(
                        action.cpu().numpy()
                    )
                    reward_tensor = _as_float_tensor(reward, device)
                    done_array = np.logical_or(termination, truncation)

                    buffer.add(
                        obs=obs_tensor,
                        actions=action,
                        raw_actions=raw_action,
                        logprobs=logprob,
                        rewards=reward_tensor,
                        dones=next_done,
                        values=value,
                    )

                    next_done = _as_float_tensor(done_array, device)
                    global_step += num_envs

                    for episode_return, episode_length in _extract_episode_metrics(
                        infos
                    ):
                        train_returns.append(episode_return)
                        train_lengths.append(episode_length)

                with torch.no_grad():
                    next_obs_tensor = _normalize_obs(
                        next_obs,
                        device=device,
                        obs_rms=obs_rms,
                        clip=obs_clip,
                        update=False,
                    )
                    _, _, _, next_value, _, _ = model.get_action_and_value(
                        next_obs_tensor,
                    )
                    advantages, returns = compute_gae(
                        rewards=buffer.rewards,
                        dones=buffer.dones,
                        values=buffer.values,
                        next_value=next_value,
                        next_done=next_done,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                    )
                buffer.set_advantages_and_returns(advantages, returns)

                diagnostics = ppo_update(
                    model=model,
                    optimizer=optimizer,
                    batch=buffer.flatten(),
                    config=_ppo_config(config, current_batch_size),
                )

                eval_metrics = _evaluation_summary([])
                should_eval = (
                    next_eval_step is not None
                    and global_step >= next_eval_step
                    and num_eval_episodes > 0
                )
                if should_eval:
                    eval_metrics = evaluate_policy(
                        model=model,
                        env_id=env_id,
                        seed=seed + 10_000,
                        episodes=num_eval_episodes,
                        device=device,
                        obs_rms=obs_rms,
                        obs_clip=obs_clip,
                    )
                    mean_return = eval_metrics["eval/mean_return"]
                    if mean_return > best_eval_return:
                        best_eval_return = mean_return
                        save_checkpoint(
                            checkpoint_dir / "best.pt",
                            model=model,
                            optimizer=optimizer,
                            global_step=global_step,
                            config=config,
                            obs_rms=obs_rms,
                            extra=_checkpoint_extra(best_eval_return),
                        )
                    while next_eval_step is not None and global_step >= next_eval_step:
                        next_eval_step += eval_interval

                should_checkpoint = (
                    next_checkpoint_step is not None
                    and global_step >= next_checkpoint_step
                )
                if should_checkpoint:
                    save_checkpoint(
                        checkpoint_dir / "latest.pt",
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        config=config,
                        obs_rms=obs_rms,
                        extra=_checkpoint_extra(best_eval_return),
                    )
                    save_checkpoint(
                        checkpoint_dir / f"step_{global_step}.pt",
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        config=config,
                        obs_rms=obs_rms,
                        extra=_checkpoint_extra(best_eval_return),
                    )
                    while (
                        next_checkpoint_step is not None
                        and global_step >= next_checkpoint_step
                    ):
                        next_checkpoint_step += checkpoint_interval

                elapsed = max(time.time() - start_time, 1e-9)
                train_return = train_returns[-1] if train_returns else float("nan")
                train_length = train_lengths[-1] if train_lengths else float("nan")
                metrics = {
                    "global_step": global_step,
                    "update": update,
                    "train/episodic_return": train_return,
                    "train/episodic_length": train_length,
                    "eval/mean_return": eval_metrics["eval/mean_return"],
                    "eval/std_return": eval_metrics["eval/std_return"],
                    "eval/min_return": eval_metrics["eval/min_return"],
                    "eval/max_return": eval_metrics["eval/max_return"],
                    "loss/policy_loss": diagnostics["policy_loss"],
                    "loss/value_loss": diagnostics["value_loss"],
                    "loss/entropy": diagnostics["entropy"],
                    "loss/approx_kl": diagnostics["approx_kl"],
                    "loss/clip_fraction": diagnostics["clip_fraction"],
                    "loss/explained_variance": diagnostics["explained_variance"],
                    "policy/action_mean": float(buffer.actions.mean().detach().cpu()),
                    "policy/action_std": float(buffer.actions.std().detach().cpu()),
                    "policy/log_std_mean": float(model.log_std.mean().detach().cpu()),
                    "optimizer/learning_rate": diagnostics["learning_rate"],
                    "system/fps": int(global_step / elapsed),
                }
                _append_metrics(writer, metrics_file, metrics)

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            config=config,
            obs_rms=obs_rms,
            extra=_checkpoint_extra(best_eval_return),
        )
        save_checkpoint(
            checkpoint_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            config=config,
            obs_rms=obs_rms,
            extra=_checkpoint_extra(best_eval_return),
        )

        return {
            "run_dir": run_dir,
            "global_step": global_step,
            "best_eval_return": best_eval_return,
            "metrics_path": metrics_path,
            "latest_checkpoint": checkpoint_dir / "latest.pt",
        }
    finally:
        envs.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on continuous control.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total training timesteps.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run directory name.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint.",
    )
    return parser.parse_args()


def _apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = dict(config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.resume is not None:
        config["resume"] = str(args.resume)
    return config


def main() -> None:
    args = parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    summary = run_training(config)
    print(
        "Training completed: "
        f"run_dir={summary['run_dir']}, "
        f"global_step={summary['global_step']}, "
        f"best_eval_return={summary['best_eval_return']:.3f}"
    )


if __name__ == "__main__":
    main()
