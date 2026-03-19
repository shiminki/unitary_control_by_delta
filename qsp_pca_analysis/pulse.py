"""Pulse construction: convert QSP phases to pulse schedule."""

import math

import numpy as np
import pandas as pd
import torch

from .constants import OMEGA_MHZ, DELTA_0_MHZ


def phi_to_pulse_df(
    phi,
    Omega_mhz: float = OMEGA_MHZ,
    Delta_0_mhz: float = DELTA_0_MHZ,
) -> pd.DataFrame:
    """Convert QSP phases to a pulse schedule DataFrame.

    Parameters
    ----------
    phi : 1-D array-like of K+1 phase values.
    Omega_mhz : Rabi frequency in MHz.
    Delta_0_mhz : Maximum detuning range in MHz.

    Returns
    -------
    DataFrame with columns "t (us)", "Omega_x (2pi MHz)",
    "Omega_y (2pi MHz)", "Omega_z (2pi MHz)".
    """
    if isinstance(phi, torch.Tensor):
        phi_np = phi.detach().cpu().numpy()
    else:
        phi_np = np.asarray(phi, dtype=float)

    Omega_ang = 2 * math.pi * Omega_mhz
    Delta_0_ang = 2 * math.pi * Delta_0_mhz
    tau_us = math.pi / (2 * Delta_0_ang)

    t_rows, hx_rows, hy_rows, hz_rows = [], [], [], []
    for i, pv in enumerate(phi_np):
        t_rows.append(np.abs(pv) / Omega_ang)
        hx_rows.append(Omega_mhz * np.sign(pv) if np.abs(pv) > 1e-15 else 0.0)
        hy_rows.append(0.0)
        hz_rows.append(0.0)
        if i < len(phi_np) - 1:
            t_rows.append(tau_us)
            hx_rows.append(0.0)
            hy_rows.append(0.0)
            hz_rows.append(Delta_0_mhz)

    return pd.DataFrame({
        "t (us)": t_rows,
        "Omega_x (2pi MHz)": hx_rows,
        "Omega_y (2pi MHz)": hy_rows,
        "Omega_z (2pi MHz)": hz_rows,
    })
