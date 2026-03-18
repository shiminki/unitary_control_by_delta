"""Inference pipeline for the joint neural network pulse generator.

Loads a pretrained JointPulseGeneratorNet and generates pulse schedules
from alpha_vals. Designed for experimentalists to quickly produce pulse_df
from rotation angle specifications.

Usage:
    from neural_network_optimization.inference import load_model, generate_pulse

    model = load_model(Omega_mhz=80.0, K=70)
    pulse_df = generate_pulse(model, alpha_vals=[0.5, 1.0, 0.3, 0.8])
    print(pulse_df)

    # Or compute fidelity:
    from neural_network_optimization.inference import compute_fidelity
    fid = compute_fidelity(model, alpha_vals=[0.5, 1.0, 0.3, 0.8])
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from neural_network_optimization.model import (
    JointPulseGeneratorNet,
    phi_to_pulse_df,
    get_weight_path,
    get_runtime_from_phi,
    DEFAULT_K,
    DEFAULT_OMEGA_MHZ,
)
from single_pulse_optimization_QSP.qsp_fit_x_rotation import fidelity_from_pulse


def load_model(
    Omega_mhz: float = DEFAULT_OMEGA_MHZ,
    K: int = DEFAULT_K,
    weight_dir: str = "neural_network_optimization",
    device: str = "cpu",
    **model_kwargs,
) -> JointPulseGeneratorNet:
    """Load a pretrained JointPulseGeneratorNet.

    Parameters
    ----------
    Omega_mhz : Rabi frequency in MHz (must match training).
    K : QSP degree (must match training).
    weight_dir : Directory containing weights/ subfolder.
    device : "cpu" or "cuda".
    **model_kwargs : Additional keyword arguments passed to JointPulseGeneratorNet
                     (e.g. hidden_dim, num_layers, n_freq if non-default).

    Returns
    -------
    JointPulseGeneratorNet in eval mode.

    Raises
    ------
    FileNotFoundError if pretrained weights are not found.
    """
    weight_path = get_weight_path(weight_dir, Omega_mhz, K)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"No pretrained weights found at {weight_path}. "
            f"Run the training pipeline first:\n"
            f"  python -m neural_network_optimization.train "
            f"--Omega_mhz {Omega_mhz} --K {K}"
        )

    net = JointPulseGeneratorNet(K=K, Omega_mhz=Omega_mhz, **model_kwargs)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    net.to(device)
    return net


def generate_pulse(
    model: JointPulseGeneratorNet,
    alpha_vals,
) -> pd.DataFrame:
    """Generate pulse schedule from alpha_vals.

    Parameters
    ----------
    model : JointPulseGeneratorNet (pretrained, in eval mode).
    alpha_vals : list, array, or tensor of N_peaks rotation angles in radians.

    Returns
    -------
    pulse_df : DataFrame with columns
        "t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"
    """
    alpha_t = _to_alpha_tensor(alpha_vals, model.N_peaks)

    model.eval()
    with torch.no_grad():
        phi = model(alpha_t.to(torch.float64)).squeeze(0)

    return phi_to_pulse_df(phi, model.Omega_mhz, model.Delta_0_mhz)


def generate_phi(
    model: JointPulseGeneratorNet,
    alpha_vals,
) -> torch.Tensor:
    """Generate raw QSP phases from alpha_vals.

    Returns (K+1,) tensor of phase values.
    """
    alpha_t = _to_alpha_tensor(alpha_vals, model.N_peaks)

    model.eval()
    with torch.no_grad():
        phi = model(alpha_t.to(torch.float64)).squeeze(0)

    return phi


def compute_fidelity(
    model: JointPulseGeneratorNet,
    alpha_vals,
    sample_size: int = 5000,
) -> float:
    """Compute average gate fidelity for given alpha_vals.

    Uses fidelity_from_pulse for an independent, pulse-based evaluation
    (no QSP internals, matches experimental simulation).

    Parameters
    ----------
    model : JointPulseGeneratorNet (pretrained).
    alpha_vals : list, array, or tensor of N_peaks rotation angles in radians.
    sample_size : Detuning samples per peak for Monte Carlo averaging.

    Returns
    -------
    Average gate fidelity (float in [0, 1]).
    """
    pulse_df = generate_pulse(model, alpha_vals)
    delta_mhz = np.array(model.delta_centers_mhz)

    if isinstance(alpha_vals, torch.Tensor):
        alpha_arr = alpha_vals.detach().cpu().numpy()
    else:
        alpha_arr = np.asarray(alpha_vals, dtype=float)

    return fidelity_from_pulse(
        pulse_df, delta_mhz, alpha_arr,
        model.robustness_window_mhz, sample_size=sample_size,
    )


def compute_runtime(
    model: JointPulseGeneratorNet,
    alpha_vals,
) -> float:
    """Compute total QSP sequence runtime in microseconds.

    Returns
    -------
    Runtime in microseconds.
    """
    phi = generate_phi(model, alpha_vals)
    return get_runtime_from_phi(phi, model.Omega_ang, model.Delta_0_ang)


def _to_alpha_tensor(alpha_vals, N_peaks: int) -> torch.Tensor:
    """Convert alpha_vals to (1, N_peaks) float64 tensor."""
    if isinstance(alpha_vals, (list, tuple)):
        alpha_t = torch.tensor(alpha_vals, dtype=torch.float64).unsqueeze(0)
    elif isinstance(alpha_vals, np.ndarray):
        alpha_t = torch.from_numpy(alpha_vals).to(torch.float64).unsqueeze(0)
    elif isinstance(alpha_vals, torch.Tensor):
        alpha_t = alpha_vals.to(torch.float64)
        if alpha_t.ndim == 1:
            alpha_t = alpha_t.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported type for alpha_vals: {type(alpha_vals)}")

    if alpha_t.shape[-1] != N_peaks:
        raise ValueError(
            f"alpha_vals has {alpha_t.shape[-1]} elements, expected {N_peaks}"
        )
    return alpha_t
