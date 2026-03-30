"""Evaluation and independent fidelity verification."""

import importlib.util
import os
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from single_pulse_optimization_QSP.qsp_fit_x_rotation import fidelity_from_pulse

from .model import PulseGeneratorNet
from .pulse import phi_to_pulse_df


def evaluate_nn(
    net: PulseGeneratorNet,
    alpha_test: torch.Tensor,
    sample_size: int = 2000,
) -> Dict[str, list]:
    """Evaluate the trained network at given alpha values using fidelity_from_pulse.

    Returns dict with keys "alpha", "fidelity", "pulse_dfs".
    """
    net.eval()
    fidelities = []
    pulse_dfs = []

    peak_idx = net.peak_index
    n_peaks = len(net.delta_centers_mhz)
    delta_mhz_arr = np.array(net.delta_centers_mhz)

    with torch.no_grad():
        for alpha_val in alpha_test:
            av = float(alpha_val)
            alpha_t = torch.tensor([av], dtype=torch.float64)
            phi = net(alpha_t).squeeze(0)
            pdf = phi_to_pulse_df(phi, net.Omega_mhz, net.Delta_0_mhz)

            alpha_arr = np.zeros(n_peaks)
            alpha_arr[peak_idx] = av

            fid = fidelity_from_pulse(
                pdf, delta_mhz_arr, alpha_arr,
                net.robustness_window_mhz, sample_size=sample_size,
            )
            fidelities.append(fid)
            pulse_dfs.append(pdf)

    return {
        "alpha": [float(a) for a in alpha_test],
        "fidelity": fidelities,
        "pulse_dfs": pulse_dfs,
    }


def compute_fidelity_for_phi(
    phi, peak_index: int, alpha: float,
    delta_centers_mhz: list, Omega_mhz: float, Delta_0_mhz: float,
    robustness_window_mhz: float, sample_size: int = 5000,
) -> float:
    """Compute gate fidelity for a given phi vector at a given alpha."""
    pdf = phi_to_pulse_df(phi, Omega_mhz, Delta_0_mhz)
    delta_arr = np.array(delta_centers_mhz)
    alpha_arr = np.zeros(len(delta_arr))
    alpha_arr[peak_index] = alpha
    return fidelity_from_pulse(pdf, delta_arr, alpha_arr,
                               robustness_window_mhz, sample_size=sample_size)


def _get_fidelity_independent():
    """Lazily import fidelity_independent from test.py for independent verification."""
    test_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test.py"
    )
    spec = importlib.util.spec_from_file_location("test_fidelity", test_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fidelity_independent


def verify_phi_fidelities(
    phi_matrix, alpha_grid, net, sample_size=1000,
):
    """Verify fidelity of all phi vectors using independent calculation from test.py.

    For each alpha_grid[i], phi_matrix[i] should produce R_x(alpha_grid[i])
    at peak_index and identity at all other peaks.

    Returns array of fidelities (one per alpha sample).
    """
    fidelity_fn = _get_fidelity_independent()
    M = len(alpha_grid)
    fidelities = np.empty(M)

    for i in tqdm(range(M), desc="Verifying fidelity (independent)"):
        pdf = phi_to_pulse_df(phi_matrix[i], net.Omega_mhz, net.Delta_0_mhz)
        alpha_arr = [0.0] * len(net.delta_centers_mhz)
        alpha_arr[net.peak_index] = float(alpha_grid[i])
        fidelities[i] = fidelity_fn(
            pdf, net.delta_centers_mhz, alpha_arr,
            net.robustness_window_mhz, sample_size=sample_size,
        )

    return fidelities
