"""PulseNet: MLP that maps (alpha_0, ..., alpha_3) to QSP phases phi[0..K]."""

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .constants import (
    DELTA_0_MHZ,
    DELTA_CENTERS_MHZ,
    N_PEAKS,
    ROBUSTNESS_WINDOW_MHZ,
)


class PulseNet(nn.Module):
    """Deep MLP mapping N_peaks rotation angles to K+1 QSP phases.

    Signature: model(alpha) where alpha is (B, N_peaks) or (N_peaks,) real tensor;
    returns (B, K+1) phi or (K+1,) phi respectively.

    Input encoding: Fourier features [cos(k alpha_i / 2), sin(k alpha_i / 2)]
    for k = 1..n_freq, giving in_dim = N_peaks * 2 * n_freq.  The half-angle
    matches SU(2) 4pi-periodicity in alpha.
    """

    def __init__(
        self,
        Omega: float,
        K: int,
        N_peaks: int = N_PEAKS,
        hidden_dim: int = 512,
        num_layers: int = 8,
        n_freq: int = 8,
        Delta_0_mhz: float = DELTA_0_MHZ,
        robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
        delta_centers_mhz=None,
    ):
        super().__init__()
        if delta_centers_mhz is None:
            delta_centers_mhz = list(DELTA_CENTERS_MHZ)

        self.Omega_mhz = float(Omega)
        self.K = int(K)
        self.N_peaks = int(N_peaks)
        self.n_freq = int(n_freq)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.Delta_0_mhz = float(Delta_0_mhz)
        self.robustness_window_mhz = float(robustness_window_mhz)
        self.delta_centers_mhz = list(delta_centers_mhz)

        self.Omega_ang = 2 * math.pi * self.Omega_mhz
        self.Delta_0_ang = 2 * math.pi * self.Delta_0_mhz
        self.robustness_window_ang = 2 * math.pi * self.robustness_window_mhz
        self.delta_centers_ang = [2 * math.pi * d for d in self.delta_centers_mhz]

        # Stash config on the state_dict so load_model can reconstruct without
        # external metadata.  These buffers never take gradients.
        self.register_buffer("_Omega_mhz_t", torch.tensor(self.Omega_mhz, dtype=torch.float64))
        self.register_buffer("_K_t", torch.tensor(self.K, dtype=torch.long))
        self.register_buffer("_N_peaks_t", torch.tensor(self.N_peaks, dtype=torch.long))
        self.register_buffer("_n_freq_t", torch.tensor(self.n_freq, dtype=torch.long))
        self.register_buffer("_hidden_dim_t", torch.tensor(self.hidden_dim, dtype=torch.long))
        self.register_buffer("_num_layers_t", torch.tensor(self.num_layers, dtype=torch.long))
        self.register_buffer("_Delta_0_mhz_t", torch.tensor(self.Delta_0_mhz, dtype=torch.float64))
        self.register_buffer("_robust_mhz_t", torch.tensor(self.robustness_window_mhz, dtype=torch.float64))
        self.register_buffer("_delta_centers_mhz_t",
                             torch.tensor(self.delta_centers_mhz, dtype=torch.float64))

        in_dim = self.N_peaks * 2 * self.n_freq
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_dim, self.hidden_dim, dtype=torch.float64))
            layers.append(nn.SiLU())
            in_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.K + 1, dtype=torch.float64))
        self.mlp = nn.Sequential(*layers)

        # Small init so initial phi ~ 0 (identity QSP unitary at start of training).
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def encode(self, alpha: torch.Tensor) -> torch.Tensor:
        """Fourier features (cos(k alpha/2), sin(k alpha/2))_{k=1..n_freq}."""
        ks = torch.arange(
            1, self.n_freq + 1, dtype=alpha.dtype, device=alpha.device,
        )
        feats = []
        for i in range(self.N_peaks):
            angles = alpha[:, i : i + 1] * ks / 2  # (B, n_freq)
            feats.append(torch.cos(angles))
            feats.append(torch.sin(angles))
        return torch.cat(feats, dim=-1)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """(B, N_peaks) -> (B, K+1).  Accepts (N_peaks,) and returns (K+1,)."""
        squeeze_out = False
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(0)
            squeeze_out = True
        alpha = alpha.to(dtype=torch.float64)
        x = self.encode(alpha)
        phi = self.mlp(x)
        if squeeze_out:
            phi = phi.squeeze(0)
        return phi


# ─────────────────────────────────────────────────────────────────────────────
#  Pulse schedule / runtime helpers
# ─────────────────────────────────────────────────────────────────────────────

def phi_to_pulse_df(
    phi,
    Omega_mhz: float,
    Delta_0_mhz: float,
) -> pd.DataFrame:
    """Convert QSP phases to a physical pulse schedule DataFrame.

    Columns: t (us), Omega_x (2pi MHz), Omega_y (2pi MHz), Omega_z (2pi MHz).
    Each control step drives at Omega_mhz * sign(phi_j) for |phi_j|/Omega_ang us;
    between controls, free evolution at Omega_z = Delta_0_mhz for pi/(2 Delta_0_ang) us.
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
        t_rows.append(abs(pv) / Omega_ang)
        hx_rows.append(Omega_mhz * np.sign(pv) if abs(pv) > 1e-15 else 0.0)
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


def get_runtime_from_phi(phi, Omega_ang: float, Delta_0_ang: float) -> float:
    """Total pulse duration (us)."""
    if isinstance(phi, torch.Tensor):
        abs_sum = float(phi.abs().sum().item())
        K = phi.shape[-1] - 1
    else:
        phi_np = np.asarray(phi, dtype=float)
        abs_sum = float(np.abs(phi_np).sum())
        K = phi_np.shape[-1] - 1
    tau = math.pi / (2 * Delta_0_ang)
    return K * tau + abs_sum / Omega_ang


# ─────────────────────────────────────────────────────────────────────────────
#  Weight path convention
# ─────────────────────────────────────────────────────────────────────────────

def get_weight_path(weight_dir: str, Omega_mhz: float, K: int) -> str:
    import os
    return os.path.join(weight_dir, f"joint_Omega{float(Omega_mhz)}_K{int(K)}.pt")
