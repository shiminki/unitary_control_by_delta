"""Inference helpers: load trained PulseNet weights and generate pulse artefacts."""

import os
from typing import Union

import numpy as np
import pandas as pd
import torch

from .model import PulseNet, get_weight_path, get_runtime_from_phi, phi_to_pulse_df
from .physics import fidelity_from_pulse


def _as_alpha_tensor(alpha, N_peaks: int, dtype=torch.float64) -> torch.Tensor:
    if isinstance(alpha, torch.Tensor):
        t = alpha.detach().to(dtype=dtype)
    else:
        t = torch.tensor(alpha, dtype=dtype)
    if t.ndim == 0:
        t = t.repeat(N_peaks)
    if t.shape[-1] != N_peaks:
        raise ValueError(f"alpha must have last-dim {N_peaks}, got {tuple(t.shape)}")
    return t


def load_model(
    Omega: float,
    K: int,
    weight_dir: str = "neural_network_optimization/weights",
    device=None,
) -> PulseNet:
    """Load a trained PulseNet from ``joint_Omega{Omega}_K{K}.pt``.

    Reads architectural buffers from the state_dict so the returned model
    matches the one that produced the checkpoint.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    path = get_weight_path(weight_dir, Omega, K)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No weights at {path}.  Train this config first with "
            f"`python -m neural_network_optimization.train --Omega {Omega} --K {K}` "
            f"or run the full scaling law."
        )
    state = torch.load(path, map_location=device, weights_only=True)

    # Reconstruct the model with matching hyperparameters from the buffers.
    def _buf(name, default):
        return state[name].item() if name in state else default

    N_peaks = int(_buf("_N_peaks_t", 4))
    n_freq = int(_buf("_n_freq_t", 8))
    hidden_dim = int(_buf("_hidden_dim_t", 512))
    num_layers = int(_buf("_num_layers_t", 8))
    Delta_0_mhz = float(_buf("_Delta_0_mhz_t", 200.0))
    robust_mhz = float(_buf("_robust_mhz_t", 10.0))
    if "_delta_centers_mhz_t" in state:
        delta_centers = state["_delta_centers_mhz_t"].tolist()
    else:
        from .constants import DELTA_CENTERS_MHZ
        delta_centers = list(DELTA_CENTERS_MHZ)

    net = PulseNet(
        Omega=Omega, K=K, N_peaks=N_peaks,
        hidden_dim=hidden_dim, num_layers=num_layers, n_freq=n_freq,
        Delta_0_mhz=Delta_0_mhz, robustness_window_mhz=robust_mhz,
        delta_centers_mhz=delta_centers,
    ).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def generate_phi(model: PulseNet, alpha) -> torch.Tensor:
    """Return phi of shape (K+1,) for a length-N_peaks alpha."""
    dev = next(model.parameters()).device
    a = _as_alpha_tensor(alpha, model.N_peaks).to(dev).unsqueeze(0)  # (1, N_peaks)
    model.eval()
    with torch.no_grad():
        phi = model(a).squeeze(0)
    return phi.detach().cpu()


def generate_pulse(model: PulseNet, alpha) -> pd.DataFrame:
    phi = generate_phi(model, alpha)
    return phi_to_pulse_df(phi, model.Omega_mhz, model.Delta_0_mhz)


def compute_fidelity(model: PulseNet, alpha, sample_size: int = 5000) -> float:
    """Physical-basis fidelity (averaged over +/- robustness window per peak)."""
    a = _as_alpha_tensor(alpha, model.N_peaks).cpu().numpy()
    pulse_df = generate_pulse(model, alpha)
    return fidelity_from_pulse(
        pulse_df, np.asarray(model.delta_centers_mhz), a,
        robustness_window_mhz=model.robustness_window_mhz,
        sample_size=sample_size,
    )


def compute_runtime(model: PulseNet, alpha) -> float:
    phi = generate_phi(model, alpha)
    return get_runtime_from_phi(phi, model.Omega_ang, model.Delta_0_ang)
