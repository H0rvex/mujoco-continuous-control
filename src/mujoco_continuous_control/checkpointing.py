from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _copy_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Copy nested metadata into a checkpoint-friendly plain dict."""

    copied: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            copied[str(key)] = _copy_mapping(value)
        elif isinstance(value, Path):
            copied[str(key)] = str(value)
        else:
            copied[str(key)] = value
    return copied


def _state_dict_payload(value: Any | None, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _copy_mapping(value)
    if hasattr(value, "state_dict"):
        state_dict = value.state_dict()
        if not isinstance(state_dict, Mapping):
            msg = f"{name}.state_dict() must return a mapping."
            raise TypeError(msg)
        return _copy_mapping(state_dict)
    msg = f"{name} must be None, a mapping, or expose state_dict()."
    raise TypeError(msg)


def _load_torch_checkpoint(
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


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: Mapping[str, Any],
    obs_rms: Any | None = None,
    extra: Mapping[str, Any] | None = None,
    reward_normalizer: Any | None = None,
) -> dict[str, Any]:
    """Save a PPO training checkpoint and return the serialized payload."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = _copy_mapping(config)
    extra_dict = _copy_mapping(extra) if extra is not None else {}
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(global_step),
        "config": config_dict,
        "obs_rms": _state_dict_payload(obs_rms, "obs_rms"),
        "reward_normalizer": _state_dict_payload(
            reward_normalizer,
            "reward_normalizer",
        ),
        "env_id": config_dict.get("env_id"),
        "seed": config_dict.get("seed"),
        "best_eval_return": extra_dict.get(
            "best_eval_return",
            config_dict.get("best_eval_return"),
        ),
    }
    checkpoint.update(extra_dict)

    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` and optionally ``optimizer``."""

    checkpoint_path = Path(path)
    checkpoint = _load_torch_checkpoint(checkpoint_path, map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
