"""
Neural network pulse generator for detuning-selective unitary control.

Given a detuning peak index i, this module trains a neural network that maps
rotation angle theta -> QSP phases phi[0..K], such that the resulting pulse
applies Rx(theta) at peak i and identity at all other peaks.

After training, PCA analysis extracts the effective degrees of freedom and
basis functions b_j(t) with amplitude functions A_j(theta).

Usage:
    # Run tests first, then train for peak 0 and do PCA
    python nn_pulse_generator.py --peak_index 0 --steps 5000

    # Run only tests
    python nn_pulse_generator.py --test_only

    # Run acceptance test (slow, ~3 min)
    python nn_pulse_generator.py --test_only --acceptance_test
"""

import argparse
import math
import os
import shutil
import sys
import unittest
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    build_qsp_unitary,
    fidelity_from_pulse,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

DELTA_CENTERS_MHZ = [-100.0, -32.0, 32.0, 100.0]
N_PEAKS = len(DELTA_CENTERS_MHZ)
OMEGA_MHZ = 80.0
DELTA_0_MHZ = 200.0
ROBUSTNESS_WINDOW_MHZ = 10.0
DEFAULT_K = 70
EPS = 0.1


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════

class PulseGeneratorNet(nn.Module):
    """Neural network mapping rotation angle theta -> QSP phases phi[0..K].

    Parameters
    ----------
    peak_index : int
        Which detuning peak (0..N-1) this network targets.
    K : int
        QSP sequence degree (phi has K+1 elements).
    hidden_dim : int
        Width of hidden layers.
    num_layers : int
        Number of hidden layers.
    n_freq : int
        Number of Fourier frequencies for the input encoding.
        Input dim = 2 * n_freq: [cos(theta), sin(theta), cos(2*theta), ...].
    delta_centers_mhz : list
        Detuning peak centers in MHz.
    Omega_mhz : float
        Rabi frequency in MHz.
    Delta_0_mhz : float
        Maximum detuning range in MHz.
    robustness_window_mhz : float
        Half-width of robustness window in MHz.
    """

    def __init__(
        self,
        peak_index: int,
        K: int = DEFAULT_K,
        hidden_dim: int = 128,
        num_layers: int = 4,
        n_freq: int = 8,
        delta_centers_mhz: list = None,
        Omega_mhz: float = OMEGA_MHZ,
        Delta_0_mhz: float = DELTA_0_MHZ,
        robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
    ):
        super().__init__()
        if delta_centers_mhz is None:
            delta_centers_mhz = list(DELTA_CENTERS_MHZ)

        self.peak_index = peak_index
        self.K = K
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

        # Fourier input encoding: [cos(k*theta), sin(k*theta)] for k=1..n_freq
        in_dim = 2 * n_freq
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

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Map rotation angle(s) to QSP phase vectors.

        Parameters
        ----------
        theta : (B,) tensor of rotation angles in radians.

        Returns
        -------
        (B, K+1) tensor of QSP phase values.
        """
        # Fourier features: [cos(theta), sin(theta), cos(2*theta), ...]
        ks = torch.arange(1, self.n_freq + 1, dtype=theta.dtype, device=theta.device)
        # theta: (B,) -> (B, 1) * (n_freq,) -> (B, n_freq)
        angles = theta.unsqueeze(-1) * ks.unsqueeze(0)
        x = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, 2*n_freq)
        return self.mlp(x)  # (B, K+1)


# ═══════════════════════════════════════════════════════════════════════════
#  Pulse construction
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  Pre-sampled detunings & loss
# ═══════════════════════════════════════════════════════════════════════════

def presample_detunings(
    peak_index: int,
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
    peak_mask : (N*S,) bool tensor, True for samples at the target peak.
    """
    n_peaks = len(delta_centers_ang)
    delta_list = []
    mask_list = []

    for j in range(n_peaks):
        center = delta_centers_ang[j]
        half_w = robustness_window_ang
        jitter = (2.0 * torch.rand(samples_per_peak, dtype=torch.float64, device=device) - 1.0) * half_w
        delta_s = (center + jitter).clamp(-Delta_0_ang, Delta_0_ang)
        delta_list.append(delta_s)
        mask_list.append(torch.full((samples_per_peak,), j == peak_index, device=device))

    return torch.cat(delta_list), torch.cat(mask_list)


def compute_batch_loss(
    net: PulseGeneratorNet,
    theta_batch: torch.Tensor,
    delta_all: torch.Tensor,
    peak_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute u_00 MSE loss for a batch of target rotation angles.

    Uses pre-sampled, fixed detuning values (no stochastic resampling).
    Computes loss directly from QSP unitary u_00 element—no gradient
    regularization overhead, so only 1 build_qsp_unitary call per theta.

    Parameters
    ----------
    net : PulseGeneratorNet
    theta_batch : (B,) tensor of target rotation angles.
    delta_all : (N*S,) pre-sampled detuning values.
    peak_mask : (N*S,) bool, True for target-peak samples.

    Returns
    -------
    Scalar loss tensor (differentiable).
    """
    B = theta_batch.shape[0]
    device = theta_batch.device

    phi_batch = net(theta_batch)  # (B, K+1)

    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

    for b in range(B):
        phi_b = phi_batch[b]  # (K+1,)

        # Alpha for target peak = theta_b, for other peaks = 0
        alpha_all = torch.where(
            peak_mask,
            theta_batch[b].expand_as(delta_all),
            torch.zeros_like(delta_all),
        )

        U = build_qsp_unitary(phi_b, delta_all, net.Delta_0_ang, net.Omega_ang)
        pred = U[:, 0, 0]  # (N*S,)
        target = (torch.cos(alpha_all / 2) - 1j * torch.sin(alpha_all / 2)).to(torch.complex128)

        loss_b = ((pred - target).abs() ** 2).mean()
        total_loss = total_loss + loss_b

    return total_loss / B


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════

def train_nn(
    peak_index: int,
    K: int = DEFAULT_K,
    hidden_dim: int = 128,
    num_layers: int = 4,
    n_freq: int = 8,
    steps: int = 5000,
    lr: float = 5e-3,
    batch_size: int = 8,
    samples_per_peak: int = 128,
    delta_centers_mhz: list = None,
    Omega_mhz: float = OMEGA_MHZ,
    Delta_0_mhz: float = DELTA_0_MHZ,
    robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
    device: str = "cpu",
    verbose: bool = True,
    out_dir: str = "nn_pulse_output",
    resample_every: int = 0,
) -> PulseGeneratorNet:
    """Train a PulseGeneratorNet for a given peak index.

    Parameters
    ----------
    resample_every : int
        If > 0, resample detuning points every this many steps (0 = never).

    Returns the trained network.
    """
    if delta_centers_mhz is None:
        delta_centers_mhz = list(DELTA_CENTERS_MHZ)
    torch.set_default_dtype(torch.float64)

    net = PulseGeneratorNet(
        peak_index=peak_index, K=K, hidden_dim=hidden_dim,
        num_layers=num_layers, n_freq=n_freq,
        delta_centers_mhz=delta_centers_mhz,
        Omega_mhz=Omega_mhz, Delta_0_mhz=Delta_0_mhz,
        robustness_window_mhz=robustness_window_mhz,
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Pre-sample detunings once (like GRAPE)
    delta_all, peak_mask = presample_detunings(
        peak_index, net.delta_centers_ang, net.robustness_window_ang,
        net.Delta_0_ang, samples_per_peak, device,
    )

    best_loss = float("inf")
    best_state = None

    iterator = tqdm(range(1, steps + 1), desc=f"Training peak {peak_index}") if verbose else range(1, steps + 1)

    for step in iterator:
        # Optionally resample detunings for generalization
        if resample_every > 0 and step > 1 and step % resample_every == 0:
            delta_all, peak_mask = presample_detunings(
                peak_index, net.delta_centers_ang, net.robustness_window_ang,
                net.Delta_0_ang, samples_per_peak, device,
            )
        
        # Ensuring it covers theta = 0 and 2pi
        theta_batch = (torch.rand(batch_size, dtype=torch.float64, device=device) * (1 + 2 * EPS) - EPS) \
            * 2 * math.pi
        loss = compute_batch_loss(net, theta_batch, delta_all, peak_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        sched.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.4e}", "best": f"{best_loss:.4e}"})

    if best_state is not None:
        net.load_state_dict(best_state)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, f"peak{peak_index}_K{K}.pt"))

    return net


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_nn(
    net: PulseGeneratorNet,
    theta_test: torch.Tensor,
    sample_size: int = 2000,
) -> Dict[str, list]:
    """Evaluate the trained network at given theta values using fidelity_from_pulse.

    Returns dict with keys "theta", "fidelity", "pulse_dfs".
    """
    net.eval()
    fidelities = []
    pulse_dfs = []

    peak_idx = net.peak_index
    n_peaks = len(net.delta_centers_mhz)
    delta_mhz_arr = np.array(net.delta_centers_mhz)

    with torch.no_grad():
        for theta_val in theta_test:
            tv = float(theta_val)
            theta_t = torch.tensor([tv], dtype=torch.float64)
            phi = net(theta_t).squeeze(0)
            pdf = phi_to_pulse_df(phi, net.Omega_mhz, net.Delta_0_mhz)

            alpha_arr = np.zeros(n_peaks)
            alpha_arr[peak_idx] = tv

            fid = fidelity_from_pulse(
                pdf, delta_mhz_arr, alpha_arr,
                net.robustness_window_mhz, sample_size=sample_size,
            )
            fidelities.append(fid)
            pulse_dfs.append(pdf)

    return {
        "theta": [float(t) for t in theta_test],
        "fidelity": fidelities,
        "pulse_dfs": pulse_dfs,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PCA analysis
# ═══════════════════════════════════════════════════════════════════════════

def pca_analysis(
    net: PulseGeneratorNet,
    M: int = 200,
    explained_var_threshold: float = 0.999,
    out_dir: str = "nn_pulse_output",
) -> Dict:
    """Perform PCA on the phi vectors generated across theta in [0, 2pi].

    Parameters
    ----------
    net : trained PulseGeneratorNet
    M : number of theta samples
    explained_var_threshold : cumulative variance threshold for selecting components

    Returns
    -------
    Dict with keys: "theta_grid", "phi_matrix", "mean_phi",
    "principal_components", "singular_values", "explained_variance_ratio",
    "n_effective_dof", "amplitudes", "amplitude_fits"
    """
    net.eval()
    theta_grid = torch.linspace(0, 2 * math.pi, M, dtype=torch.float64)

    with torch.no_grad():
        phi_matrix = net(theta_grid).cpu().numpy()  # (M, K+1)

    mean_phi = phi_matrix.mean(axis=0)
    phi_centered = phi_matrix - mean_phi

    _, S, Vt = np.linalg.svd(phi_centered, full_matrices=False)

    var = S ** 2
    total_var = var.sum()
    explained_ratio = var / total_var if total_var > 0 else var
    cumulative = np.cumsum(explained_ratio)

    n_dof = int(np.searchsorted(cumulative, explained_var_threshold) + 1)
    n_dof = min(n_dof, len(S))

    # Principal components: (K+1, n_dof)
    pc = Vt[:n_dof, :].T

    # Amplitudes: projection onto PCs
    amplitudes = phi_centered @ pc  # (M, n_dof)

    # Fit each A_j(theta) with a Fourier series
    theta_np = theta_grid.numpy()
    amplitude_fits = []
    for j in range(n_dof):
        amp_j = amplitudes[:, j]
        n_fourier = min(10, M // 4)
        design = [np.ones(M)]
        for k in range(1, n_fourier + 1):
            design.append(np.cos(k * theta_np))
            design.append(np.sin(k * theta_np))
        design = np.column_stack(design)
        coeffs, _, _, _ = np.linalg.lstsq(design, amp_j, rcond=None)
        residual = float(np.mean((amp_j - design @ coeffs) ** 2))
        amplitude_fits.append({
            "coeffs": coeffs,
            "n_fourier": n_fourier,
            "residual": residual,
        })

    os.makedirs(out_dir, exist_ok=True)
    _plot_pca_results(
        theta_np, phi_matrix, S, explained_ratio, cumulative,
        n_dof, amplitudes, amplitude_fits, peak_index=net.peak_index,
        out_dir=out_dir, explained_var_threshold=explained_var_threshold,
    )

    result = {
        "theta_grid": theta_np,
        "phi_matrix": phi_matrix,
        "mean_phi": mean_phi,
        "principal_components": pc,
        "singular_values": S,
        "explained_variance_ratio": explained_ratio,
        "n_effective_dof": n_dof,
        "amplitudes": amplitudes,
        "amplitude_fits": amplitude_fits,
    }

    save_pca_csv(result, net.peak_index, out_dir)

    return result


def save_pca_csv(pca_result: Dict, peak_index: int, out_dir: str):
    """Save basis functions b_j^i(t) and Fourier coefficients of A_j(theta) as CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    n_dof = pca_result["n_effective_dof"]
    mean_phi = pca_result["mean_phi"]
    fits = pca_result["amplitude_fits"]

    # --- Basis functions CSV ---
    # Columns: phase_index, mean_phi, b_0, b_1, ..., b_{n_dof-1}
    K_plus_1 = pc.shape[0]
    basis_data = {"phase_index": np.arange(K_plus_1), "mean_phi": mean_phi}
    for j in range(n_dof):
        basis_data[f"b_{j}"] = pc[:, j]
    df_basis = pd.DataFrame(basis_data)
    df_basis.to_csv(
        os.path.join(out_dir, f"basis_functions_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )

    # --- Fourier coefficients CSV ---
    # One row per component j, columns: component, a_0, a_cos1, a_sin1, ..., residual
    rows = []
    for j in range(n_dof):
        coeffs = fits[j]["coeffs"]
        n_fourier = fits[j]["n_fourier"]
        row = {"component": j, "a_0": coeffs[0]}
        for k in range(1, n_fourier + 1):
            row[f"a_cos{k}"] = coeffs[2 * k - 1]
            row[f"a_sin{k}"] = coeffs[2 * k]
        row["residual"] = fits[j]["residual"]
        rows.append(row)
    df_coeffs = pd.DataFrame(rows)
    df_coeffs.to_csv(
        os.path.join(out_dir, f"fourier_coefficients_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )


def _plot_pca_results(
    theta_grid, phi_matrix, S, explained_ratio, cumulative,
    n_dof, amplitudes, amplitude_fits, peak_index, out_dir,
    explained_var_threshold=0.999,
):
    """Generate PCA analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Singular value spectrum
    ax = axes[0, 0]
    ax.semilogy(S / S[0], "o-", markersize=4)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title(f"Singular Value Spectrum (peak {peak_index})")
    ax.grid(True, alpha=0.3)

    # 2. Residual unexplained variance (1 - cumulative)
    ax = axes[0, 1]
    residual_var = 1.0 - cumulative
    residual_var = np.clip(residual_var, a_min=1e-16, a_max=None)
    ax.semilogy(residual_var, "o-", markersize=4)
    ax.axhline(
        y=1.0 - explained_var_threshold, color="r", linestyle="--", alpha=0.5,
        label=f"{(1 - explained_var_threshold) * 100:.1f}% threshold",
    )
    ax.axvline(x=n_dof - 1, color="g", linestyle="--", alpha=0.5, label=f"d={n_dof}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("1 − cumulative explained variance")
    ax.set_title(f"Residual Variance (Effective DOF = {n_dof})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Amplitude functions A_j(theta)
    ax = axes[1, 0]
    for j in range(min(n_dof, 5)):
        ax.plot(theta_grid / math.pi, amplitudes[:, j], label=f"A_{j}")
        fit = amplitude_fits[j]
        n_fourier = fit["n_fourier"]
        design = [np.ones(len(theta_grid))]
        for k in range(1, n_fourier + 1):
            design.append(np.cos(k * theta_grid))
            design.append(np.sin(k * theta_grid))
        design = np.column_stack(design)
        ax.plot(theta_grid / math.pi, design @ fit["coeffs"], "--", alpha=0.7)
    ax.set_xlabel(r"$\theta / \pi$")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude functions (solid=data, dashed=Fourier fit)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Sample phi trajectories
    ax = axes[1, 1]
    n_show = min(5, phi_matrix.shape[0])
    indices = np.linspace(0, phi_matrix.shape[0] - 1, n_show, dtype=int)
    for idx in indices:
        theta_val = theta_grid[idx]
        ax.plot(phi_matrix[idx], label=f"{theta_val/math.pi:.2f}pi", alpha=0.7)
    ax.set_xlabel("Phase index j")
    ax.set_ylabel("phi_j")
    ax.set_title("Sample phi vectors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"PCA Analysis for Peak {peak_index}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_peak{peak_index}.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Analytical reconstruction from PCA
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_phi_analytical(pca_result: Dict, theta: float) -> np.ndarray:
    """Reconstruct phi(theta) from the PCA decomposition.

    phi(theta) = mean_phi + sum_j A_j(theta) * pc_j

    where A_j(theta) is evaluated from its Fourier fit coefficients.
    """
    mean_phi = pca_result["mean_phi"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    fits = pca_result["amplitude_fits"]
    n_dof = pca_result["n_effective_dof"]

    phi_recon = mean_phi.copy()
    for j in range(n_dof):
        coeffs = fits[j]["coeffs"]
        n_fourier = fits[j]["n_fourier"]
        val = coeffs[0]
        for k in range(1, n_fourier + 1):
            val += coeffs[2 * k - 1] * np.cos(k * theta)
            val += coeffs[2 * k] * np.sin(k * theta)
        phi_recon += val * pc[:, j]

    return phi_recon


def _pulse_df_to_time_series(pulse_df: pd.DataFrame):
    """Extract cumulative time edges and field values from a pulse_df.

    Returns (t_edges_ns, hx, hz) where t_edges_ns has length N+1 and
    hx/hz have length N.
    """
    ts = pulse_df["t (us)"].to_numpy(dtype=float) * 1000  # ns
    hx = pulse_df["Omega_x (2pi MHz)"].to_numpy(dtype=float)
    hz = pulse_df["Omega_z (2pi MHz)"].to_numpy(dtype=float)
    t_edges = np.concatenate(([0.0], np.cumsum(ts)))
    return t_edges, hx, hz


def _compute_matrix_elements(phi, Delta_0_ang, Omega_ang, n_delta=1024):
    """Compute physical-basis u_00 and u_01 vs detuning.

    Returns (delta_mhz, u00, u01).
    """
    if isinstance(phi, np.ndarray):
        phi_t = torch.tensor(phi, dtype=torch.float64)
    else:
        phi_t = phi.detach().clone().to(torch.float64)

    delta_range = torch.linspace(-Delta_0_ang, Delta_0_ang, n_delta,
                                 dtype=torch.float64)
    U = build_qsp_unitary(phi_t, delta_range, Delta_0_ang, Omega_ang)

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    U = H @ U @ H  # QSP -> physical basis

    delta_mhz = (delta_range / (2 * math.pi)).numpy()
    u00 = U[:, 0, 0].detach().cpu().numpy()
    u01 = U[:, 0, 1].detach().cpu().numpy()
    return delta_mhz, u00, u01


# ═══════════════════════════════════════════════════════════════════════════
#  Feature 1: Comparative pulse parameter plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparative_pulses(
    net: PulseGeneratorNet,
    pca_result: Dict,
    theta_vals: list = None,
    out_dir: str = "nn_pulse_output",
):
    """Plot H_x(t) and H_z(t) for NN output vs PCA-analytical reconstruction.

    For each theta in theta_vals, produces a row with two subplots
    (H_x on left, H_z on right) comparing the actual NN pulse with
    the analytical PCA-reconstructed pulse.
    """
    if theta_vals is None:
        theta_vals = [math.pi / 4, math.pi / 2, math.pi]

    net.eval()
    os.makedirs(out_dir, exist_ok=True)

    n_theta = len(theta_vals)
    fig, axes = plt.subplots(n_theta, 2, figsize=(16, 4 * n_theta), squeeze=False)

    for row, theta in enumerate(theta_vals):
        with torch.no_grad():
            phi_nn = net(torch.tensor([theta], dtype=torch.float64)).squeeze(0)
        pdf_nn = phi_to_pulse_df(phi_nn, net.Omega_mhz, net.Delta_0_mhz)
        t_nn, hx_nn, hz_nn = _pulse_df_to_time_series(pdf_nn)

        phi_pca = reconstruct_phi_analytical(pca_result, theta)
        pdf_pca = phi_to_pulse_df(phi_pca, net.Omega_mhz, net.Delta_0_mhz)
        t_pca, hx_pca, hz_pca = _pulse_df_to_time_series(pdf_pca)

        theta_label = f"{theta/math.pi:.2f}"

        # H_x
        ax = axes[row, 0]
        ax.step(t_nn, np.r_[hx_nn, hx_nn[-1]], where="post",
                label="NN (actual)", linewidth=1.5)
        ax.step(t_pca, np.r_[hx_pca, hx_pca[-1]], where="post",
                label="PCA (analytical)", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(r"$\Omega_x(t)$ (2$\pi$ MHz)")
        ax.set_ylim(-1.1 * net.Omega_mhz, 1.1 * net.Omega_mhz)
        ax.set_title(rf"$\theta = {theta_label}\pi$  —  $\Omega_x(t)$")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == n_theta - 1:
            ax.set_xlabel("Time (ns)")

        # H_z
        ax = axes[row, 1]
        ax.step(t_nn, np.r_[hz_nn, hz_nn[-1]], where="post",
                label="NN (actual)", linewidth=1.5)
        ax.step(t_pca, np.r_[hz_pca, hz_pca[-1]], where="post",
                label="PCA (analytical)", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(r"$\Omega_z(t)$ (2$\pi$ MHz)")
        ax.set_ylim(-10, net.Delta_0_mhz + 20)
        ax.set_title(rf"$\theta = {theta_label}\pi$  —  $\Omega_z(t)$")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == n_theta - 1:
            ax.set_xlabel("Time (ns)")

    plt.suptitle(f"Pulse Comparison: NN vs Analytical (Peak {net.peak_index})",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,
                f"comparative_pulses_peak{net.peak_index}.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Fidelity helper
# ═══════════════════════════════════════════════════════════════════════════

def compute_fidelity_for_phi(
    phi, peak_index: int, theta: float,
    delta_centers_mhz: list, Omega_mhz: float, Delta_0_mhz: float,
    robustness_window_mhz: float, sample_size: int = 5000,
) -> float:
    """Compute gate fidelity for a given phi vector at a given theta."""
    pdf = phi_to_pulse_df(phi, Omega_mhz, Delta_0_mhz)
    delta_arr = np.array(delta_centers_mhz)
    alpha_arr = np.zeros(len(delta_arr))
    alpha_arr[peak_index] = theta
    return fidelity_from_pulse(pdf, delta_arr, alpha_arr,
                               robustness_window_mhz, sample_size=sample_size)


# ═══════════════════════════════════════════════════════════════════════════
#  Feature 2: Comparative matrix element plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparative_matrix_elements(
    net: PulseGeneratorNet,
    pca_result: Dict,
    theta_vals: list = None,
    out_dir: str = "nn_pulse_output",
):
    """Plot Re(u_00) and Im(u_01) vs detuning for NN vs PCA-analytical.

    For each theta, shows the matrix elements of the physical-basis unitary
    across all detunings, comparing the actual NN pulse with the analytical
    PCA-reconstructed pulse.  Target windows are overlaid.

    Returns a dict mapping theta -> {"fidelity_nn": float, "fidelity_pca": float}.
    """
    if theta_vals is None:
        theta_vals = [math.pi / 4, math.pi / 2, math.pi]

    net.eval()
    os.makedirs(out_dir, exist_ok=True)

    n_theta = len(theta_vals)
    fig, axes = plt.subplots(n_theta, 1, figsize=(10, 5 * n_theta), squeeze=False)

    sigma_mhz = net.robustness_window_mhz
    delta_centers = net.delta_centers_mhz
    peak_idx = net.peak_index

    fidelity_results = {}

    for row, theta in enumerate(theta_vals):
        ax = axes[row, 0]

        with torch.no_grad():
            phi_nn = net(torch.tensor([theta], dtype=torch.float64)).squeeze(0)
        dm_nn, u00_nn, u01_nn = _compute_matrix_elements(
            phi_nn, net.Delta_0_ang, net.Omega_ang)

        phi_pca = reconstruct_phi_analytical(pca_result, theta)
        dm_pca, u00_pca, u01_pca = _compute_matrix_elements(
            phi_pca, net.Delta_0_ang, net.Omega_ang)

        # Compute fidelities
        fid_nn = compute_fidelity_for_phi(
            phi_nn, peak_idx, theta, delta_centers,
            net.Omega_mhz, net.Delta_0_mhz, sigma_mhz)
        fid_pca = compute_fidelity_for_phi(
            phi_pca, peak_idx, theta, delta_centers,
            net.Omega_mhz, net.Delta_0_mhz, sigma_mhz)
        fidelity_results[theta] = {"fidelity_nn": fid_nn, "fidelity_pca": fid_pca}

        ax.plot(dm_nn, u00_nn.real, color="C0", linewidth=1.2,
                linestyle="--", label="Re(u_00) NN")
        ax.plot(dm_nn, u01_nn.imag, color="C1", linewidth=1.2,
                linestyle="--", label="Im(u_01) NN")
        ax.plot(dm_pca, u00_pca.real, color="C2", linewidth=1.2,
                alpha=0.7, label="Re(u_00) PCA")
        ax.plot(dm_pca, u01_pca.imag, color="C3", linewidth=1.2,
                alpha=0.7, label="Im(u_01) PCA")

        for j, dc in enumerate(delta_centers):
            alpha_j = theta if j == peak_idx else 0.0
            ax.hlines(y=np.cos(alpha_j / 2),
                      xmin=dc - sigma_mhz, xmax=dc + sigma_mhz,
                      colors="red", linestyles="dotted", linewidth=1.5,
                      label="target Re(u_00)" if (row == 0 and j == 0) else None)
            ax.hlines(y=-np.sin(alpha_j / 2),
                      xmin=dc - sigma_mhz, xmax=dc + sigma_mhz,
                      colors="green", linestyles="dotted", linewidth=1.5,
                      label="target Im(u_01)" if (row == 0 and j == 0) else None)
            ax.axvspan(dc - sigma_mhz, dc + sigma_mhz, color="gray", alpha=0.1)

        theta_label = f"{theta/math.pi:.2f}"
        ax.set_title(
            rf"$\theta = {theta_label}\pi$  —  "
            rf"Fidelity: NN = {fid_nn:.4f}, PCA = {fid_pca:.4f}")
        ax.set_ylabel("Matrix Element")
        ax.set_ylim(-1.3, 1.3)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        if row == n_theta - 1:
            ax.set_xlabel(r"Detuning $\delta$ (MHz)")

    plt.suptitle(
        f"Matrix Element Comparison: NN vs Analytical (Peak {peak_idx})",
        fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_dir, f"comparative_matrix_elements_peak{peak_idx}.png"), dpi=150)
    plt.close()

    return fidelity_results


# ═══════════════════════════════════════════════════════════════════════════
#  Feature 3: Analytical form output
# ═══════════════════════════════════════════════════════════════════════════

def format_analytical_form(
    pca_result: Dict,
    coeff_threshold: float = 1e-3,
    fidelity_results: Dict = None,
) -> str:
    """Format the PCA decomposition as a human-readable analytical expression.

    Parameters
    ----------
    fidelity_results : dict, optional
        Mapping theta -> {"fidelity_nn": float, "fidelity_pca": float}.
        If provided, fidelity comparison is appended.

    Returns a multi-line string describing:
      H_c^i(t; theta) = phi_mean(t) + sum_j A_j(theta) * b_j^i(t)
    with A_j as Fourier series and b_j as weight vectors over phase indices.
    """
    n_dof = pca_result["n_effective_dof"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    mean_phi = pca_result["mean_phi"]
    fits = pca_result["amplitude_fits"]
    sv = pca_result["singular_values"]
    ev = pca_result["explained_variance_ratio"]

    K = pc.shape[0] - 1
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYTICAL DECOMPOSITION")
    lines.append(
        f"  H_c^i(t; theta) = phi_mean(t)"
        f" + sum_{{j=0}}^{{{n_dof-1}}} A_j(theta) * b_j(t)")
    lines.append(f"  Effective DOF: d = {n_dof}  (K = {K})")
    lines.append(f"  Explained variance: {sum(ev[:n_dof])*100:.2f}%")
    lines.append("=" * 70)

    nonzero_mean = np.where(np.abs(mean_phi) > coeff_threshold)[0]
    lines.append(f"\nphi_mean: {len(nonzero_mean)} / {K+1} nonzero entries")
    lines.append(f"  ||phi_mean|| = {np.linalg.norm(mean_phi):.4f}")

    for j in range(n_dof):
        lines.append(f"\n{'─' * 50}")
        lines.append(
            f"Component j = {j}  (singular value = {sv[j]:.4f}, "
            f"variance = {ev[j]*100:.2f}%)")
        lines.append("─" * 50)

        coeffs = fits[j]["coeffs"]
        n_fourier = fits[j]["n_fourier"]
        residual = fits[j]["residual"]

        terms = []
        if abs(coeffs[0]) > coeff_threshold:
            terms.append(f"{coeffs[0]:+.4f}")
        for k in range(1, n_fourier + 1):
            c_cos = coeffs[2 * k - 1]
            c_sin = coeffs[2 * k]
            if abs(c_cos) > coeff_threshold:
                terms.append(f"{c_cos:+.4f} cos({k}*theta)")
            if abs(c_sin) > coeff_threshold:
                terms.append(f"{c_sin:+.4f} sin({k}*theta)")

        if not terms:
            a_str = "0"
        else:
            a_str = " ".join(terms)
            if a_str.startswith("+"):
                a_str = a_str[1:]

        lines.append(f"  A_{j}(theta) = {a_str}")
        lines.append(f"  Fourier fit residual: {residual:.2e}")

        bj = pc[:, j]
        sorted_idx = np.argsort(np.abs(bj))[::-1]
        top_n = min(8, len(sorted_idx))

        lines.append(f"  b_{j}(t): dominant phase indices (top {top_n}):")
        for idx in sorted_idx[:top_n]:
            if abs(bj[idx]) < coeff_threshold:
                break
            lines.append(f"    phi[{idx:3d}] weight = {bj[idx]:+.4f}")

        lines.append(f"  ||b_{j}|| = {np.linalg.norm(bj):.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("RECONSTRUCTION FORMULA")
    lines.append("  For a given theta, the QSP phases are:")
    lines.append(
        f"    phi_j = mean_phi_j"
        f" + sum_{{k=0}}^{{{n_dof-1}}} A_k(theta) * b_k[j]")
    lines.append("  Then the pulse schedule is built via phi_to_pulse_df(phi).")
    lines.append("=" * 70)

    if fidelity_results:
        lines.append("\n" + "=" * 70)
        lines.append("FIDELITY COMPARISON: NN (original) vs PCA (reconstructed)")
        lines.append("=" * 70)
        for theta in sorted(fidelity_results.keys()):
            fid = fidelity_results[theta]
            theta_label = f"{theta/math.pi:.2f}*pi"
            lines.append(
                f"  theta = {theta_label:>10s}  |  "
                f"NN = {fid['fidelity_nn']:.6f}  |  "
                f"PCA = {fid['fidelity_pca']:.6f}")
        lines.append("=" * 70)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestPulseGeneratorNetShapes(unittest.TestCase):
    """Verify forward pass produces correct output shapes."""

    def test_single_input(self):
        net = PulseGeneratorNet(peak_index=0, K=10, n_freq=4)
        theta = torch.tensor([1.0], dtype=torch.float64)
        self.assertEqual(net(theta).shape, (1, 11))

    def test_batch_input(self):
        net = PulseGeneratorNet(peak_index=0, K=30, n_freq=8)
        theta = torch.rand(16, dtype=torch.float64) * 2 * math.pi
        self.assertEqual(net(theta).shape, (16, 31))

    def test_output_dtype(self):
        net = PulseGeneratorNet(peak_index=0, K=5, n_freq=4)
        out = net(torch.tensor([0.5], dtype=torch.float64))
        self.assertEqual(out.dtype, torch.float64)


class TestPhiToPulseDf(unittest.TestCase):
    """Verify pulse_df structure from phi_to_pulse_df."""

    def test_row_count(self):
        K = 4
        phi = np.array([0.5, -0.3, 0.1, 0.8, -0.2])
        self.assertEqual(len(phi_to_pulse_df(phi)), 2 * K + 1)

    def test_control_rows_have_zero_omega_z(self):
        pdf = phi_to_pulse_df(np.array([0.5, -0.3, 0.1]))
        for i in [0, 2, 4]:
            self.assertAlmostEqual(pdf["Omega_z (2pi MHz)"].iloc[i], 0.0)

    def test_signal_rows_have_zero_omega_x(self):
        pdf = phi_to_pulse_df(np.array([0.5, -0.3, 0.1]))
        for i in [1, 3]:
            self.assertAlmostEqual(pdf["Omega_x (2pi MHz)"].iloc[i], 0.0)
            self.assertAlmostEqual(pdf["Omega_z (2pi MHz)"].iloc[i], DELTA_0_MHZ)

    def test_control_duration(self):
        phi_val = 1.5
        pdf = phi_to_pulse_df(np.array([phi_val]))
        expected_t = abs(phi_val) / (2 * math.pi * OMEGA_MHZ)
        self.assertAlmostEqual(pdf["t (us)"].iloc[0], expected_t, places=12)

    def test_omega_x_sign(self):
        pdf = phi_to_pulse_df(np.array([0.5, -0.3]))
        self.assertGreater(pdf["Omega_x (2pi MHz)"].iloc[0], 0)
        self.assertLess(pdf["Omega_x (2pi MHz)"].iloc[2], 0)

    def test_from_tensor(self):
        phi = torch.tensor([0.5, -0.3, 0.1], dtype=torch.float64)
        self.assertEqual(len(phi_to_pulse_df(phi)), 5)


class TestGradientFlow(unittest.TestCase):
    """Verify gradients flow from loss through the network."""

    def test_gradients_exist(self):
        torch.manual_seed(42)
        net = PulseGeneratorNet(peak_index=0, K=4, hidden_dim=16, num_layers=2, n_freq=4)
        theta = torch.tensor([1.0], dtype=torch.float64)

        delta_all, peak_mask = presample_detunings(
            0, net.delta_centers_ang, net.robustness_window_ang,
            net.Delta_0_ang, samples_per_peak=32,
        )
        loss = compute_batch_loss(net, theta, delta_all, peak_mask)
        loss.backward()

        for name, param in net.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")
            self.assertFalse(torch.all(param.grad == 0), f"{name} all-zero gradient")


class TestTrainingReducesLoss(unittest.TestCase):
    """Verify that a short training run reduces loss."""

    def test_loss_decreases(self):
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)

        net = PulseGeneratorNet(peak_index=0, K=4, hidden_dim=32, num_layers=2, n_freq=4)
        opt = torch.optim.Adam(net.parameters(), lr=5e-3)

        delta_all, peak_mask = presample_detunings(
            0, net.delta_centers_ang, net.robustness_window_ang,
            net.Delta_0_ang, samples_per_peak=64,
        )

        theta_eval = torch.tensor([0.0, math.pi], dtype=torch.float64)
        initial_loss = compute_batch_loss(net, theta_eval, delta_all, peak_mask).item()

        for _ in range(200):
            theta_batch = torch.rand(4, dtype=torch.float64) * 2 * math.pi
            loss = compute_batch_loss(net, theta_batch, delta_all, peak_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()

        final_loss = compute_batch_loss(net, theta_eval, delta_all, peak_mask).item()
        self.assertLess(final_loss, initial_loss,
                        f"Loss didn't decrease: {initial_loss:.4e} -> {final_loss:.4e}")


class TestPulseDfCompatibility(unittest.TestCase):
    """Verify NN-generated pulse_df works with fidelity_from_pulse."""

    def test_fidelity_returns_valid_float(self):
        torch.manual_seed(42)
        net = PulseGeneratorNet(peak_index=0, K=10, n_freq=4)

        with torch.no_grad():
            phi = net(torch.tensor([math.pi / 2], dtype=torch.float64)).squeeze(0)

        pdf = phi_to_pulse_df(phi, net.Omega_mhz, net.Delta_0_mhz)
        delta_arr = np.array(net.delta_centers_mhz)
        alpha_arr = np.zeros(len(delta_arr))
        alpha_arr[net.peak_index] = math.pi / 2

        fid = fidelity_from_pulse(pdf, delta_arr, alpha_arr,
                                  net.robustness_window_mhz, sample_size=500)
        self.assertIsInstance(fid, float)
        self.assertGreaterEqual(fid, 0.0)
        self.assertLessEqual(fid, 1.0 + 1e-6)


class TestPCADimensions(unittest.TestCase):
    """Verify PCA output has correct shapes."""

    def test_pca_shapes(self):
        torch.manual_seed(42)
        K = 5
        net = PulseGeneratorNet(peak_index=0, K=K, hidden_dim=16, num_layers=2, n_freq=4)
        M = 50
        out_dir = "test_pca_tmp"

        result = pca_analysis(net, M=M, out_dir=out_dir)

        self.assertEqual(result["phi_matrix"].shape, (M, K + 1))
        self.assertEqual(result["mean_phi"].shape, (K + 1,))
        self.assertEqual(result["theta_grid"].shape, (M,))

        n_dof = result["n_effective_dof"]
        self.assertEqual(result["principal_components"].shape, (K + 1, n_dof))
        self.assertEqual(result["amplitudes"].shape, (M, n_dof))
        self.assertGreater(n_dof, 0)
        self.assertLessEqual(n_dof, K + 1)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)


class TestHighFidelitySingleTarget(unittest.TestCase):
    """Train on a single (peak, theta) and verify fidelity >= 98%.

    Uses fixed detuning samples and direct u_00 loss (no gradient
    regularization) with aggressive LR, matching GRAPE convergence.
    """

    def test_single_target_fidelity(self):
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)

        K = 30
        peak_idx = 2  # delta = +32 MHz
        target_theta = math.pi

        # Smaller NN for single-point memorization (faster convergence)
        net = PulseGeneratorNet(
            peak_index=peak_idx, K=K,
            hidden_dim=64, num_layers=2, n_freq=8,
        )

        # Pre-sample detunings (fixed, like GRAPE)
        delta_all, peak_mask = presample_detunings(
            peak_idx, net.delta_centers_ang, net.robustness_window_ang,
            net.Delta_0_ang, samples_per_peak=128,
        )

        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=8000)

        # Train at fixed theta=pi for 8000 steps (matching GRAPE budget)
        theta_batch = torch.tensor([target_theta], dtype=torch.float64)
        for _ in range(8000):
            loss = compute_batch_loss(net, theta_batch, delta_all, peak_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            sched.step()

        # Evaluate via independent fidelity_from_pulse
        net.eval()
        with torch.no_grad():
            phi = net(torch.tensor([target_theta], dtype=torch.float64)).squeeze(0)
        pdf = phi_to_pulse_df(phi, net.Omega_mhz, net.Delta_0_mhz)

        delta_arr = np.array(net.delta_centers_mhz)
        alpha_arr = np.zeros(len(delta_arr))
        alpha_arr[peak_idx] = target_theta

        fid = fidelity_from_pulse(pdf, delta_arr, alpha_arr,
                                  net.robustness_window_mhz, sample_size=5000)
        self.assertGreaterEqual(
            fid, 0.98,
            f"Single-target fidelity {fid:.4f} < 0.98 after 8000 steps",
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_tests(verbosity=2):
    """Run fast unit tests. Returns True if all pass."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestPulseGeneratorNetShapes,
        TestPhiToPulseDf,
        TestGradientFlow,
        TestTrainingReducesLoss,
        TestPulseDfCompatibility,
        TestPCADimensions,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return unittest.TextTestRunner(verbosity=verbosity).run(suite).wasSuccessful()


def run_acceptance_test(verbosity=2):
    """Run the high-fidelity acceptance test (slow, ~3 min)."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestHighFidelitySingleTarget)
    return unittest.TextTestRunner(verbosity=verbosity).run(suite).wasSuccessful()


def main():
    ap = argparse.ArgumentParser(description="NN pulse generator for QSP")
    ap.add_argument("--peak_index", type=int, default=0, help="Detuning peak index (0..3)")
    ap.add_argument("--K", type=int, default=DEFAULT_K, help="QSP degree")
    ap.add_argument("--steps", type=int, default=5000, help="Training steps")
    ap.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=8, help="Theta batch size")
    ap.add_argument("--hidden_dim", type=int, default=128, help="NN hidden dimension")
    ap.add_argument("--num_layers", type=int, default=4, help="NN hidden layers")
    ap.add_argument("--n_freq", type=int, default=8, help="Fourier input frequencies")
    ap.add_argument("--out_dir", type=str, default="nn_pulse_output", help="Output directory")
    ap.add_argument("--test_only", action="store_true", help="Run tests only")
    ap.add_argument("--skip_tests", action="store_true", help="Skip tests")
    ap.add_argument("--acceptance_test", action="store_true", help="Run acceptance test (slow)")
    ap.add_argument("--M_pca", type=int, default=200, help="PCA theta samples")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    if not args.skip_tests:
        print("=" * 60)
        print("Running unit tests...")
        print("=" * 60)
        if not run_tests():
            print("Unit tests FAILED. Aborting.")
            sys.exit(1)
        print("\nAll unit tests passed.\n")

    if args.acceptance_test:
        print("=" * 60)
        print("Running acceptance test (~3 min)...")
        print("=" * 60)
        if not run_acceptance_test():
            print("Acceptance test FAILED.")
            sys.exit(1)
        print("\nAcceptance test passed.\n")

    if args.test_only:
        return

    # Train
    print("=" * 60)
    print(f"Training NN for peak {args.peak_index}, K={args.K}, steps={args.steps}")
    print("=" * 60)

    net = train_nn(
        peak_index=args.peak_index,
        K=args.K,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_freq=args.n_freq,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating trained network...")
    print("=" * 60)

    theta_eval = torch.linspace(0, 2 * math.pi, 9, dtype=torch.float64)
    results = evaluate_nn(net, theta_eval)
    for tv, fid in zip(results["theta"], results["fidelity"]):
        print(f"  theta = {tv/math.pi:.2f}*pi  |  fidelity = {fid:.6f}")

    # PCA
    print("\n" + "=" * 60)
    print("Running PCA analysis...")
    print("=" * 60)

    pca_result = pca_analysis(net, M=args.M_pca, out_dir=args.out_dir)
    print(f"  Effective DOF: {pca_result['n_effective_dof']}")
    print(f"  Singular values (top 10): {pca_result['singular_values'][:10].round(4)}")
    print(f"  Explained variance (top 10): {pca_result['explained_variance_ratio'][:10].round(4)}")

    # Feature 1: Comparative pulse plots
    compare_thetas = [math.pi / 4, math.pi / 2, math.pi]
    print("\nGenerating comparative pulse plots...")
    plot_comparative_pulses(net, pca_result, compare_thetas, args.out_dir)

    # Feature 2: Comparative matrix element plots (also computes fidelities)
    print("Generating comparative matrix element plots...")
    fidelity_results = plot_comparative_matrix_elements(
        net, pca_result, compare_thetas, args.out_dir)

    # Feature 3: Analytical form (with fidelity comparison)
    print("\n")
    analytical_str = format_analytical_form(pca_result, fidelity_results=fidelity_results)
    print(analytical_str)

    # Save analytical form to file
    analytical_path = os.path.join(args.out_dir, f"analytical_form_peak{args.peak_index}.txt")
    with open(analytical_path, "w") as f:
        f.write(analytical_str)

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
