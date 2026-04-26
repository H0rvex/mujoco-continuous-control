from __future__ import annotations

import torch

from mujoco_continuous_control.normalization import RunningMeanStd


def test_running_mean_std_uses_float64_stats() -> None:
    rms = RunningMeanStd(shape=(3,))
    rms.update(torch.randn(16, 3, dtype=torch.float32))

    assert rms.mean.dtype == torch.float64
    assert rms.var.dtype == torch.float64
    assert rms.count.dtype == torch.float64


def test_normalized_obs_has_stable_values() -> None:
    rms = RunningMeanStd(shape=(2,))
    observations = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )

    normalized = rms.update_and_normalize(observations)

    assert normalized.shape == observations.shape
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-3)
    assert torch.allclose(
        normalized.std(dim=0, unbiased=False),
        torch.ones(2),
        atol=1e-3,
    )
    assert torch.isfinite(normalized).all()


def test_normalize_clips_extreme_values() -> None:
    rms = RunningMeanStd(shape=(1,))
    rms.update(torch.zeros(32, 1))

    normalized = rms.normalize(torch.tensor([[1000.0], [-1000.0]]), clip=2.0)

    assert torch.equal(normalized, torch.tensor([[2.0], [-2.0]]))


def test_stats_save_and_load_for_checkpoint_round_trip() -> None:
    rms = RunningMeanStd(shape=(3,))
    rms.update(torch.randn(64, 3) * 2.0 + 5.0)

    loaded = RunningMeanStd(shape=(3,))
    loaded.load_state_dict(rms.state_dict())

    assert torch.allclose(loaded.mean, rms.mean)
    assert torch.allclose(loaded.var, rms.var)
    assert torch.allclose(loaded.count, rms.count)


def test_loaded_frozen_stats_do_not_update_during_evaluation() -> None:
    rms = RunningMeanStd(shape=(2,))
    rms.update(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    frozen = RunningMeanStd.from_state_dict(rms.state_dict(), frozen=True)
    original_mean = frozen.mean.clone()
    original_var = frozen.var.clone()
    original_count = frozen.count.clone()

    normalized = frozen.update_and_normalize(torch.tensor([[100.0, 200.0]]))

    assert torch.isfinite(normalized).all()
    assert torch.equal(frozen.mean, original_mean)
    assert torch.equal(frozen.var, original_var)
    assert torch.equal(frozen.count, original_count)
    assert not frozen.training
