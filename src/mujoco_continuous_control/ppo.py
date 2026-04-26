from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


def _config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    return config.get(key, default)


def _explained_variance(y_pred: Tensor, y_true: Tensor) -> Tensor:
    variance = torch.var(y_true)
    if variance == 0:
        return torch.zeros((), device=y_true.device, dtype=y_true.dtype)
    return 1.0 - torch.var(y_true - y_pred) / variance


def _mean_diagnostics(history: list[dict[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    return {
        key: sum(item[key] for item in history) / len(history)
        for key in history[0]
    }


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Any,
    config: Mapping[str, Any],
) -> dict[str, float | int]:
    """Run PPO updates over a flattened rollout batch."""

    clip_coef = float(_config_value(config, "clip_coef", 0.2))
    ent_coef = float(_config_value(config, "ent_coef", 0.0))
    vf_coef = float(_config_value(config, "vf_coef", 0.5))
    max_grad_norm = float(_config_value(config, "max_grad_norm", 0.5))
    update_epochs = int(_config_value(config, "update_epochs", 1))
    minibatch_size = int(_config_value(config, "minibatch_size", batch.obs.shape[0]))
    normalize_advantages = bool(_config_value(config, "normalize_advantages", True))
    clip_vloss = bool(_config_value(config, "clip_vloss", True))
    target_kl = _config_value(config, "target_kl", None)
    target_kl = None if target_kl is None else float(target_kl)

    batch_size = batch.obs.shape[0]
    if minibatch_size < 1:
        msg = f"minibatch_size must be >= 1, got {minibatch_size}"
        raise ValueError(msg)
    if update_epochs < 1:
        msg = f"update_epochs must be >= 1, got {update_epochs}"
        raise ValueError(msg)

    advantages = batch.advantages
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    indices = torch.arange(batch_size, device=batch.obs.device)
    diagnostics_history: list[dict[str, float]] = []
    stop_early = False

    for _ in range(update_epochs):
        permutation = indices[torch.randperm(batch_size, device=batch.obs.device)]
        for start in range(0, batch_size, minibatch_size):
            mb_inds = permutation[start : start + minibatch_size]

            _, new_logprob, entropy, new_value, _, _ = model.get_action_and_value(
                batch.obs[mb_inds],
                action=batch.raw_actions[mb_inds],
            )
            old_logprob = batch.logprobs[mb_inds]
            old_value = batch.values[mb_inds]
            mb_advantages = advantages[mb_inds]
            mb_returns = batch.returns[mb_inds]

            logratio = new_logprob - old_logprob
            ratio = logratio.exp()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio,
                1.0 - clip_coef,
                1.0 + clip_coef,
            )
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            if clip_vloss:
                v_loss_unclipped = (new_value - mb_returns).pow(2)
                v_clipped = old_value + torch.clamp(
                    new_value - old_value,
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                value_loss = 0.5 * (new_value - mb_returns).pow(2).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
                explained_variance = _explained_variance(new_value, mb_returns)

            diagnostics_history.append(
                {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_loss.detach().cpu()),
                    "approx_kl": float(approx_kl.detach().cpu()),
                    "old_approx_kl": float(old_approx_kl.detach().cpu()),
                    "clip_fraction": float(clip_fraction.detach().cpu()),
                    "explained_variance": float(explained_variance.detach().cpu()),
                }
            )

            if target_kl is not None and approx_kl > target_kl:
                stop_early = True
                break

        if stop_early:
            break

    diagnostics = _mean_diagnostics(diagnostics_history)
    learning_rate = optimizer.param_groups[0]["lr"]
    diagnostics["learning_rate"] = float(learning_rate)
    diagnostics["num_updates"] = len(diagnostics_history)
    return diagnostics
