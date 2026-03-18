"""Joint neural network model for multi-peak QSP pulse generation.

Maps alpha_vals (N_peaks rotation angles) -> QSP phases phi[0..K],
such that the resulting pulse applies Rx(alpha_i) at each detuning peak i
simultaneously.

Unlike the per-peak PulseGeneratorNet in nn_pulse_generator.py, this model
handles ALL peaks jointly with a single forward pass.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    build_qsp_unitary,
    fidelity_from_pulse,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

DELTA_CENTERS_MHZ = [-100.0, -32.0, 32.0, 100.0]
N_PEAKS = len(DELTA_CENTERS_MHZ)
DELTA_0_MHZ = 200.0
ROBUSTNESS_WINDOW_MHZ = 10.0
DEFAULT_K = 70
DEFAULT_OMEGA_MHZ = 80.0
EPS = 0.1


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════

class JointPulseGeneratorNet(nn.Module):
    """Neural network mapping alpha_vals -> QSP phases phi[0..K].

    Input : alpha_vals (B, N_peaks) — rotation angles in radians for each peak.
    Output: phi        (B, K+1)    — QSP phase values.

    Uses Fourier encoding: for each peak i, [cos(k*alpha_i), sin(k*alpha_i)]
    for k = 1..n_freq, giving input dim = N_peaks * 2 * n_freq.
    """

    def __init__(
        self,
        K: int = DEFAULT_K,
        N_peaks: int = N_PEAKS,
        hidden_dim: int = 256,
        num_layers: int = 6,
        n_freq: int = 8,
        delta_centers_mhz: list = None,
        Omega_mhz: float = DEFAULT_OMEGA_MHZ,
        Delta_0_mhz: float = DELTA_0_MHZ,
        robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
    ):
        super().__init__()
        if delta_centers_mhz is None:
            delta_centers_mhz = list(DELTA_CENTERS_MHZ)

        self.K = K
        self.N_peaks = N_peaks
        self.n_freq = n_freq
        self.Omega_mhz = Omega_mhz
        self.Delta_0_mhz = Delta_0_mhz
        self.robustness_window_mhz = robustness_window_mhz
        self.delta_centers_mhz = delta_centers_mhz

        # Angular units for internal computation
        self.Omega_ang = 2 * math.pi * Omega_mhz
        self.Delta_0_ang = 2 * math.pi * Delta_0_mhz
        self.robustness_window_ang = 2 * math.pi * robustness_window_mhz
        self.delta_centers_ang = [2 * math.pi * d for d in delta_centers_mhz]

        in_dim = N_peaks * 2 * n_freq
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=torch.float64))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, K + 1, dtype=torch.float64))
        self.mlp = nn.Sequential(*layers)

        # Initialize output layer small so initial phi ~ 0 (identity)
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, alpha_vals: torch.Tensor) -> torch.Tensor:
        """Map rotation angles to QSP phase vectors.

        Parameters
        ----------
        alpha_vals : (B, N_peaks) tensor of rotation angles in radians.

        Returns
        -------
        (B, K+1) tensor of QSP phase values.
        """
        ks = torch.arange(
            1, self.n_freq + 1,
            dtype=alpha_vals.dtype, device=alpha_vals.device,
        )
        features = []
        for i in range(self.N_peaks):
            angles = alpha_vals[:, i : i + 1] * ks  # (B, n_freq)
            features.append(torch.cos(angles))
            features.append(torch.sin(angles))
        x = torch.cat(features, dim=-1)  # (B, N_peaks * 2 * n_freq)
        return self.mlp(x)  # (B, K+1)


# ═══════════════════════════════════════════════════════════════════════════
#  Pulse construction
# ═══════════════════════════════════════════════════════════════════════════

def phi_to_pulse_df(
    phi,
    Omega_mhz: float = DEFAULT_OMEGA_MHZ,
    Delta_0_mhz: float = DELTA_0_MHZ,
) -> pd.DataFrame:
    """Convert QSP phases to a pulse schedule DataFrame.

    Returns DataFrame with columns:
        "t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"
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


# ═══════════════════════════════════════════════════════════════════════════
#  Pre-sampled detunings & loss
# ═══════════════════════════════════════════════════════════════════════════

def presample_detunings_joint(
    delta_centers_ang: list,
    robustness_window_ang: float,
    Delta_0_ang: float,
    samples_per_peak: int = 128,
    device: str = "cpu",
) -> tuple:
    """Pre-sample detuning values around each peak (fixed for training).

    Returns
    -------
    delta_all : (N*S,) tensor of detuning values in angular units.
    peak_ids  : (N*S,) long tensor of peak indices (which peak each sample belongs to).
    """
    delta_list = []
    peak_id_list = []
    for j, center in enumerate(delta_centers_ang):
        jitter = (
            2.0 * torch.rand(samples_per_peak, dtype=torch.float64, device=device) - 1.0
        ) * robustness_window_ang
        delta_s = (center + jitter).clamp(-Delta_0_ang, Delta_0_ang)
        delta_list.append(delta_s)
        peak_id_list.append(
            torch.full((samples_per_peak,), j, dtype=torch.long, device=device)
        )
    return torch.cat(delta_list), torch.cat(peak_id_list)


def compute_batch_loss_joint(
    net: JointPulseGeneratorNet,
    alpha_batch: torch.Tensor,
    delta_all: torch.Tensor,
    peak_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute u_00 MSE loss for a batch of alpha_vals.

    For each detuning sample near peak i, the target is Rx(alpha_i):
        u_00 = cos(alpha_i/2) - i*sin(alpha_i/2)

    Parameters
    ----------
    net         : JointPulseGeneratorNet
    alpha_batch : (B, N_peaks) rotation angles in radians.
    delta_all   : (N*S,) pre-sampled detuning values.
    peak_ids    : (N*S,) peak index for each detuning sample.

    Returns
    -------
    Scalar loss tensor (differentiable).
    """
    B = alpha_batch.shape[0]
    device = alpha_batch.device

    phi_batch = net(alpha_batch)  # (B, K+1)

    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

    for b in range(B):
        # Target alpha at each detuning sample = alpha_batch[b, peak_id]
        alpha_targets = alpha_batch[b][peak_ids]  # (N*S,)

        U = build_qsp_unitary(phi_batch[b], delta_all, net.Delta_0_ang, net.Omega_ang)
        pred = U[:, 0, 0]  # (N*S,)
        target = (
            torch.cos(alpha_targets / 2) - 1j * torch.sin(alpha_targets / 2)
        ).to(torch.complex128)

        loss_b = ((pred - target).abs() ** 2).mean()
        total_loss = total_loss + loss_b

    return total_loss / B


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset sampling
# ═══════════════════════════════════════════════════════════════════════════

def _rand_alpha(n: int, device: str) -> torch.Tensor:
    """Sample n angles uniformly in [0, 2pi) with slight boundary extension."""
    return (
        (torch.rand(n, dtype=torch.float64, device=device) * (1 + 2 * EPS) - EPS)
        * 2 * math.pi
    )


def sample_alpha_batch(
    batch_size: int,
    N_peaks: int = N_PEAKS,
    device: str = "cpu",
) -> torch.Tensor:
    """Sample alpha_vals for training using a progressive complexity dataset.

    Dataset is a union of four types, ordered by number of active peaks:

      1. One-hot  (1 active peak): 40% — teaches each peak independently.
      2. Two-peak (2 active peaks): 25% — teaches pairwise interactions.
      3. Three-peak (3 active peaks): 15% — teaches higher-order interactions.
      4. All-random (4 active peaks): 20% — covers the full 4D space using
         Sobol quasi-random sequences for uniform coverage.

    This progressive structure is far more data-efficient than a 50/50
    all-random / one-hot split because:
    - One-hot samples are the most signal-rich (each teaches one marginal).
    - Sobol sequences fill the 4D space ~uniformly, unlike pseudo-random
      which clusters and leaves gaps.

    Returns (batch_size, N_peaks) tensor.
    """
    n_onehot  = max(1, int(batch_size * 0.40))
    n_two     = max(1, int(batch_size * 0.25))
    n_three   = max(1, int(batch_size * 0.15))
    n_full    = max(0, batch_size - n_onehot - n_two - n_three)

    samples = []

    # 1. One-hot: one random peak active, rest exactly 0
    out = torch.zeros(n_onehot, N_peaks, dtype=torch.float64, device=device)
    peak_idx = torch.randint(0, N_peaks, (n_onehot,), device=device)
    out[torch.arange(n_onehot, device=device), peak_idx] = _rand_alpha(n_onehot, device)
    samples.append(out)

    # 2. Two-peak: two random peaks active, rest exactly 0
    out = torch.zeros(n_two, N_peaks, dtype=torch.float64, device=device)
    for i in range(n_two):
        peaks = torch.randperm(N_peaks, device=device)[:2]
        out[i, peaks] = _rand_alpha(2, device)
    samples.append(out)

    # 3. Three-peak: three random peaks active, rest exactly 0
    out = torch.zeros(n_three, N_peaks, dtype=torch.float64, device=device)
    for i in range(n_three):
        peaks = torch.randperm(N_peaks, device=device)[:3]
        out[i, peaks] = _rand_alpha(3, device)
    samples.append(out)

    # 4. All-active: Sobol quasi-random for uniform 4D coverage
    if n_full > 0:
        try:
            engine = torch.quasirandom.SobolEngine(
                dimension=N_peaks, scramble=True
            )
            full = engine.draw(n_full, dtype=torch.float64).to(device)
            full = (full * (1 + 2 * EPS) - EPS) * 2 * math.pi
        except Exception:
            # Fallback to pseudo-random if Sobol unavailable
            full = (
                (torch.rand(n_full, N_peaks, dtype=torch.float64, device=device)
                 * (1 + 2 * EPS) - EPS)
                * 2 * math.pi
            )
        samples.append(full)

    return torch.cat(samples, dim=0)


# ═══════════════════════════════════════════════════════════════════════════
#  Weight path helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_weight_path(out_dir: str, Omega_mhz: float, K: int) -> str:
    """Standard weight file path for a given (Omega_mhz, K) pair."""
    return os.path.join(out_dir, "weights", f"joint_Omega{Omega_mhz}_K{K}.pt")


def get_runtime_from_phi(phi: torch.Tensor, Omega_ang: float, Delta_0_ang: float) -> float:
    """Total QSP sequence runtime in microseconds."""
    tau = math.pi / (2 * Delta_0_ang)
    K = len(phi) - 1
    return K * tau + phi.abs().sum().item() / Omega_ang
