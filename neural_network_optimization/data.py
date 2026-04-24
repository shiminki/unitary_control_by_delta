"""Dataset sampling for PulseNet training/evaluation."""

import math

import torch

from .constants import ALPHA_RANGE, EPS, N_PEAKS


def sample_alpha(n: int, N_peaks: int = N_PEAKS, device=None, generator=None) -> torch.Tensor:
    """Sample (n, N_peaks) alphas IID uniform in [-EPS, 4*pi + EPS]."""
    lo, hi = ALPHA_RANGE
    u = torch.rand(n, N_peaks, dtype=torch.float64, device=device, generator=generator)
    return u * (hi - lo) + lo


def build_dataset(
    n_train: int = 65_536,
    n_eval: int = 1024,
    N_peaks: int = N_PEAKS,
    seed: int = 0,
    device=None,
):
    """Reproducible (alpha_train, alpha_eval) tensor pair."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    train = sample_alpha(n_train, N_peaks=N_peaks, device="cpu", generator=g)
    eval_ = sample_alpha(n_eval, N_peaks=N_peaks, device="cpu", generator=g)
    if device is not None:
        train = train.to(device)
        eval_ = eval_.to(device)
    return train, eval_


__all__ = ["sample_alpha", "build_dataset", "EPS", "ALPHA_RANGE"]
