"""
Gradient-based optimization of QSP phase sequences with detuning-aware control.

Goal: learn phi[0..K] so that the QSP unitary approximates Rx(alpha_i) when
detuning delta is near delta_i, within a robustness window.

Convention (QSP basis, x-tilde = Z_physical, z-tilde = X_physical):
    U = control(phi_0) signal(theta) control(phi_1) signal(theta) ... control(phi_K)
"""

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import time

import torch
import torch.nn as nn

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import pandas as pd
from tqdm import tqdm


LAMBDA_VAL = 0.2


__all__ = [
    "TrainConfig", "train", "fidelity", "fidelity_from_pulse",
    "convert_old_pulse_to_new",
    "plot_matrix_element_vs_delta",
    "get_control_runtime", "build_qsp_unitary", "delta_to_theta",
    "theta_to_delta", "signal_operator", "Rz", "bmm", "DirectPhases",
    "str_to_bool",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Matrix primitives
# ─────────────────────────────────────────────────────────────────────────────

def Rz(phi: torch.Tensor) -> torch.Tensor:
    """
    Rz(phi) = exp(-i sigma_z phi/2) = diag(e^{-i phi/2}, e^{+i phi/2}).

    phi: shape [B] or scalar tensor (real)
    returns: shape [B,2,2] complex
    """
    phi = phi.to(dtype=torch.float64)
    e0 = torch.exp(-0.5j * phi)
    e1 = torch.exp(+0.5j * phi)
    B = phi.shape[0] if phi.ndim > 0 else 1
    out = torch.zeros((B, 2, 2), dtype=torch.complex128, device=phi.device)
    out[:, 0, 0] = e0.reshape(-1)
    out[:, 1, 1] = e1.reshape(-1)
    return out


def signal_operator(theta: torch.Tensor) -> torch.Tensor:
    """
    QSP signal operator: R_x(theta) = exp(-i theta/2 sigma_x).

    Physically realized by waiting t = pi/(2*Delta_0) with Omega=0, Delta=Delta_0+delta,
    which maps delta in [-Delta_0, Delta_0] to theta in [0, pi].

    theta: shape [B] (real)
    returns: shape [B,2,2] complex
    """
    theta = theta.to(dtype=torch.float64)
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    out = torch.zeros((theta.shape[0], 2, 2), dtype=torch.complex128, device=theta.device)
    out[:, 0, 0] = c
    out[:, 1, 1] = c
    out[:, 0, 1] = -1j * s
    out[:, 1, 0] = -1j * s
    return out


def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batch matmul for [B,2,2] @ [B,2,2]."""
    return torch.bmm(A, B)


# ─────────────────────────────────────────────────────────────────────────────
#  Phase parametrization
# ─────────────────────────────────────────────────────────────────────────────

class DirectPhases(nn.Module):
    """Learn phi[0..K] directly as free parameters (unconstrained)."""
    def __init__(self, K: int, init_scale: float = 0.01):
        super().__init__()
        raw = init_scale * torch.randn(K + 1, dtype=torch.float64)
        self.raw = nn.Parameter(raw)

    def forward(self) -> torch.Tensor:
        return self.raw


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ─────────────────────────────────────────────────────────────────────────────
#  Delta <-> theta mapping
# ─────────────────────────────────────────────────────────────────────────────

def theta_to_delta(theta_tensor: torch.Tensor, Delta_0: float):
    """Convert QSP angle theta to detuning delta. Delta_0 in angular units (rad/us).

    With W(theta) = R_x(theta), theta in [0, pi], cos(theta/2) in [0, 1]:
        delta = (2/pi * theta - 1) * Delta_0
    """
    return (2 / math.pi * theta_tensor - 1) * Delta_0


def delta_to_theta(delta_tensor: torch.Tensor, Delta_0: float):
    """Convert detuning delta to QSP angle theta. Delta_0 in angular units (rad/us).

    With W(theta) = R_x(theta), theta in [0, pi], cos(theta/2) in [0, 1]:
        theta = pi/2 * (1 + delta/Delta_0)
    """
    return math.pi / 2 * (1 + delta_tensor / Delta_0)


def get_wait_time(Delta_0):
    """Signal wait time in microseconds.

    Delta_0 is in angular units (rad/us), e.g. 2*pi*200.
    With W(theta) = R_x(theta), the physical signal rotation is:
        exp(-i*(Delta_0+delta)*tau/2 * sigma_x) = R_x((Delta_0+delta)*tau)
    Setting theta_max = 2*Delta_0*tau = pi gives tau = pi/(2*Delta_0).
    """
    return math.pi / (2 * Delta_0)


# ─────────────────────────────────────────────────────────────────────────────
#  QSP unitary builder (detuning-aware control)
# ─────────────────────────────────────────────────────────────────────────────

def build_qsp_unitary(
    phi: torch.Tensor,
    delta: torch.Tensor,
    Delta_0: float,
    Omega: float,
) -> torch.Tensor:
    """
    Build the noisy QSP unitary for a batch of detuning values.

    Sequence: control(phi_0; delta) signal(theta) control(phi_1; delta) ... control(phi_K; delta)

    The control operator includes the effect of detuning:
        H_control = 1/2 * (Omega * sigma_z + delta * sigma_x)
        for time t = |phi_j| / Omega

    Parameters
    ----------
    phi : (K+1,) tensor of QSP phase angles (radians, dimensionless)
    delta : (B,) tensor of detuning values (angular units, rad/us)
    Delta_0 : maximum detuning range (angular units, rad/us)
    Omega : Rabi frequency (angular units, rad/us)

    Returns
    -------
    (B, 2, 2) complex128 unitary tensor
    """
    assert phi.ndim == 1, "phi must be shape [K+1]"
    theta = delta_to_theta(delta, Delta_0)
    K = phi.shape[0] - 1
    B = theta.shape[0]
    dev = theta.device


    def apply_control_operator(Ucur: torch.Tensor, phase: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Apply noisy control operator R_z(phase; delta).

        Physical Hamiltonian (QSP basis):
            H_c = sign(phase) * Omega/2 * sigma_z  +  delta/2 * sigma_x
        Duration: t = |phase| / Omega

        Flipping the drive direction (negative phase) reverses the
        Omega*sigma_z term but NOT the detuning delta*sigma_x term.
        """
        Omega_tensor = torch.full((B,), Omega, dtype=torch.float64, device=dev)
        norm = torch.sqrt(Omega_tensor**2 + delta**2)  # [B,]

        abs_lamb = torch.abs(phase) / (2 * Omega) * norm  # [B,], always >= 0
        sign_phase = torch.sign(phase)
        c = torch.cos(abs_lamb)
        s = torch.sin(abs_lamb)
        sin_diag = s * sign_phase * Omega_tensor / norm  # sign flips with phase
        sin_offdiag = s * delta / norm                    # sign does NOT flip

        rotation = torch.zeros((B, 2, 2), dtype=torch.complex128, device=dev)  # [B,2,2]
        rotation[:, 0, 0] = c - 1j * sin_diag
        rotation[:, 1, 1] = c + 1j * sin_diag
        rotation[:, 0, 1] = -1j * sin_offdiag
        rotation[:, 1, 0] = -1j * sin_offdiag

        return bmm(rotation, Ucur)


    I = torch.eye(2, dtype=torch.complex128, device=dev).expand(B, 2, 2).clone()
    W_signal = signal_operator(theta)  # [B,2,2]

    # Build: control(phi_0) signal control(phi_1) signal ... signal control(phi_K)
    U = apply_control_operator(I, phi[0], delta)
    for j in range(0, K):
        U = bmm(W_signal, U)
        U = apply_control_operator(U, phi[j + 1], delta)

    return U


def build_qsp_unitary_batched(
    phi: torch.Tensor,
    delta: torch.Tensor,
    Delta_0: float,
    Omega: float,
) -> torch.Tensor:
    """
    Batched QSP unitary builder — processes multiple phi vectors in parallel.

    Same physics as build_qsp_unitary, but vectorized over both a batch of
    phi vectors AND a batch of detuning values simultaneously.  Uses scalar
    element-wise ops on (B*D,) tensors instead of torch.bmm on (B*D, 2, 2),
    which eliminates per-iteration tensor allocation and enables torch.compile
    to fuse operations across loop iterations.

    Parameters
    ----------
    phi : (B, K+1) tensor of QSP phase angles (radians)
    delta : (D,) tensor of detuning values (angular units, rad/us)
    Delta_0 : maximum detuning range (angular units, rad/us)
    Omega : Rabi frequency (angular units, rad/us)

    Returns
    -------
    (B, D, 2, 2) complex128 unitary tensor
    """
    assert phi.ndim == 2, "phi must be shape [B, K+1]"
    B, Kp1 = phi.shape
    K = Kp1 - 1
    D = delta.shape[0]
    dev = delta.device
    BD = B * D

    theta = delta_to_theta(delta, Delta_0)  # (D,)

    # Expand to (B*D,) — each batch element sees all D detunings
    delta_flat = delta.unsqueeze(0).expand(B, D).reshape(BD)
    theta_flat = theta.unsqueeze(0).expand(B, D).reshape(BD)

    # Pre-compute signal operator components (constant across K loop)
    w_c = torch.cos(theta_flat / 2).to(torch.complex128)   # diagonal
    w_s = (-1j * torch.sin(theta_flat / 2))                # off-diagonal

    # Pre-compute control operator quantities (constant across K loop)
    norm = torch.sqrt(Omega ** 2 + delta_flat ** 2)  # (BD,)

    # U = Identity: track 4 scalar components as (BD,) complex128 tensors
    u00 = torch.ones(BD, dtype=torch.complex128, device=dev)
    u01 = torch.zeros(BD, dtype=torch.complex128, device=dev)
    u10 = torch.zeros(BD, dtype=torch.complex128, device=dev)
    u11 = torch.ones(BD, dtype=torch.complex128, device=dev)

    for j in range(K + 1):
        # --- Control operator: R_z(phi[:, j]; delta) @ U ---
        phase_flat = phi[:, j].unsqueeze(1).expand(B, D).reshape(BD)
        abs_lamb = torch.abs(phase_flat) / (2 * Omega) * norm
        sign_phase = torch.sign(phase_flat)
        c = torch.cos(abs_lamb).to(torch.complex128)
        s = torch.sin(abs_lamb)
        r_diag_im = s * sign_phase * Omega / norm   # imaginary part of diagonal
        r_off_im = s * delta_flat / norm             # imaginary part of off-diagonal
        # rotation: [[c - i*r_diag_im, -i*r_off_im], [-i*r_off_im, c + i*r_diag_im]]
        r00 = c - 1j * r_diag_im
        r11 = c + 1j * r_diag_im
        r_off = -1j * r_off_im
        n00 = r00 * u00 + r_off * u10
        n01 = r00 * u01 + r_off * u11
        n10 = r_off * u00 + r11 * u10
        n11 = r_off * u01 + r11 * u11
        u00, u01, u10, u11 = n00, n01, n10, n11

        if j < K:
            # --- Signal operator: W_signal @ U ---
            # W = [[w_c, w_s], [w_s, w_c]]
            n00 = w_c * u00 + w_s * u10
            n01 = w_c * u01 + w_s * u11
            n10 = w_s * u00 + w_c * u10
            n11 = w_s * u01 + w_c * u11
            u00, u01, u10, u11 = n00, n01, n10, n11

    # Reconstruct (B, D, 2, 2)
    U = torch.stack([
        torch.stack([u00, u01], dim=-1),
        torch.stack([u10, u11], dim=-1),
    ], dim=-2).reshape(B, D, 2, 2)
    return U


# ─────────────────────────────────────────────────────────────────────────────
#  Fidelity
# ─────────────────────────────────────────────────────────────────────────────

def fidelity(phi, delta_vals, alpha_vals, cfg) -> float:
    """
    Average gate fidelity F = (|Tr(U_target^dag U)|^2 + 2) / 6
    sampled over detunings within the robustness window.

    :param phi: QSP phase angles (radians, dimensionless)
    :param delta_vals: target detuning centers (angular units, rad/us)
    :param alpha_vals: target rotation angles (radians)
    :param cfg: TrainConfig
    :return: average gate fidelity (float in [0, 1])
    """
    sample_size = 5000
    idx = torch.randint(0, delta_vals.numel(), (sample_size,), device=cfg.device)
    centers = delta_vals[idx]
    half_window = cfg.robustness_window
    jitter = (2.0 * torch.rand(sample_size, device=cfg.device) - 1.0) * half_window
    delta_samples = (centers + jitter).clamp(-cfg.Delta_0, cfg.Delta_0)
    nearest_idx = (delta_samples[:, None] - delta_vals[None, :]).abs().argmin(dim=1)
    alpha_samples = alpha_vals[nearest_idx]

    U = build_qsp_unitary(phi, delta=delta_samples, Delta_0=cfg.Delta_0, Omega=cfg.Omega_max)

    # Target in QSP basis: Rz(alpha) = diag(e^{-i alpha/2}, e^{+i alpha/2})
    target_unitary = torch.zeros((sample_size, 2, 2), dtype=torch.complex128, device=cfg.device)
    target_unitary[:, 0, 0] = torch.exp(-0.5j * alpha_samples.to(torch.complex128))
    target_unitary[:, 1, 1] = torch.exp(0.5j * alpha_samples.to(torch.complex128))

    traces = torch.einsum("bii->b", target_unitary.conj().transpose(-2, -1) @ U)
    return (((traces.abs() ** 2) + 2.0) / 6.0).mean().item()


def fidelity_from_pulse(
    pulse_df: pd.DataFrame,
    delta_mhz,
    alpha_rad,
    robustness_window_mhz: float,
    sample_size: int = 5000,
) -> float:
    """
    Standalone pulse-based fidelity evaluator (experimental simulation).

    Propagates through every time step of the pulse schedule under sampled
    detunings and compares with the target gate.  All inputs use lab units
    (MHz, radians, microseconds) so this function is independent of QSP
    internals and can be called from a notebook with just a CSV.

    Physical Hamiltonian per time step (physical basis):
        H_j = (Omega_x_j_ang / 2) sigma_x  +  ((Omega_z_j_ang + delta_ang) / 2) sigma_z
    where Omega_ang = 2*pi * Omega_mhz (column values are nominal MHz),
    and delta_ang = 2*pi * delta_mhz.

    Propagator via analytic 2x2 matrix exponential:
        V_j = exp(-i H_j t_j) = cos(r) I  -  i sin(r)/r (a sigma_x + b sigma_z)
        a = pi * Omega_x_mhz * t_j,  b = pi * (Omega_z_mhz + delta_mhz) * t_j,
        r = sqrt(a^2 + b^2).

    Target: R_x(alpha_i) in physical basis.
    Fidelity: F = ( |Tr(U_target^dag U)|^2 + 2 ) / 6, averaged.

    :param pulse_df: DataFrame with columns
                     "t (us)" | ("Omega_x (2pi MHz)" or "H_x (2pi MHz)") |
                               ("Omega_z (2pi MHz)" or "H_z (2pi MHz)").
                     Column values are in nominal MHz.
    :param delta_mhz: target detuning centers in MHz (list, array, or tensor)
    :param alpha_rad: target rotation angles in radians (list, array, or tensor)
    :param robustness_window_mhz: half-width of sampling window in MHz
    :param sample_size: number of delta samples per (delta_i, alpha_i) pair
    :return: average gate fidelity (float in [0, 1])
    """
    ts = pulse_df["t (us)"].to_numpy(dtype=float)
    Omega_xs = (
        pulse_df["Omega_x (2pi MHz)"] if "Omega_x (2pi MHz)" in pulse_df.columns
        else pulse_df["H_x (2pi MHz)"]
    ).to_numpy(dtype=float)
    Omega_zs = (
        pulse_df["Omega_z (2pi MHz)"] if "Omega_z (2pi MHz)" in pulse_df.columns
        else pulse_df["H_z (2pi MHz)"]
    ).to_numpy(dtype=float)

    if isinstance(delta_mhz, torch.Tensor):
        delta_arr = delta_mhz.detach().cpu().numpy().astype(float)
    else:
        delta_arr = np.asarray(delta_mhz, dtype=float)

    if isinstance(alpha_rad, torch.Tensor):
        alpha_arr = alpha_rad.detach().cpu().numpy().astype(float)
    else:
        alpha_arr = np.asarray(alpha_rad, dtype=float)

    rng = np.random.default_rng()
    all_fidelities = []

    for delta_i, alpha_i in zip(delta_arr, alpha_arr):
        # Sample detuning uniformly around delta_i (all in MHz)
        delta_s = rng.uniform(
            delta_i - robustness_window_mhz,
            delta_i + robustness_window_mhz,
            size=sample_size,
        )  # [S]

        # Target: R_x(alpha_i) in physical basis
        c_tgt = np.cos(alpha_i / 2)
        s_tgt = np.sin(alpha_i / 2)

        # Propagate U = V_last @ ... @ V_0, stored as four scalar arrays
        u00 = np.ones(sample_size, dtype=complex)
        u01 = np.zeros(sample_size, dtype=complex)
        u10 = np.zeros(sample_size, dtype=complex)
        u11 = np.ones(sample_size, dtype=complex)

        for j in range(len(ts)):
            a = math.pi * Omega_xs[j] * ts[j]              # scalar
            b = math.pi * (Omega_zs[j] + delta_s) * ts[j]  # [S]
            r = np.sqrt(a**2 + b**2)                        # [S]
            cos_r = np.cos(r)
            sinc_r = np.where(r > 1e-15, np.sin(r) / r, 1.0)

            v00 = cos_r - 1j * b * sinc_r       # [S]
            v01 = -1j * a * sinc_r               # [S] (== v10)
            v11 = cos_r + 1j * b * sinc_r       # [S]

            # V @ U  (2x2 manual matmul, v01 == v10)
            new_u00 = v00 * u00 + v01 * u10
            new_u01 = v00 * u01 + v01 * u11
            new_u10 = v01 * u00 + v11 * u10
            new_u11 = v01 * u01 + v11 * u11
            u00, u01, u10, u11 = new_u00, new_u01, new_u10, new_u11

        # Tr(U^dag @ R_x) where R_x = [[c, -is], [-is, c]]
        traces = (
            np.conj(u00) * c_tgt
            + np.conj(u10) * (-1j * s_tgt)
            + np.conj(u01) * (-1j * s_tgt)
            + np.conj(u11) * c_tgt
        )
        all_fidelities.append((np.abs(traces) ** 2 + 2.0) / 6.0)

    return float(np.concatenate(all_fidelities).mean())


# ─────────────────────────────────────────────────────────────────────────────
#  Old-to-new pulse conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_old_pulse_to_new(
    pulse_df_old: pd.DataFrame,
    delta_mhz,
    alpha_rad,
    Omega_max_mhz: float = 80.0,
    Delta_0_mhz: float = 200.0,
    robustness_window_mhz: float = 10.0,
    steps: int = 8000,
    lr: float = 5e-2,
    verbose: bool = True,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Convert a pulse CSV from the old convention (W_old = R_x(2*theta)) to the
    current convention (W_new = R_x(theta)) by retraining the QSP phases.

    The old code had two issues that prevent a simple tau-doubling fix:
      1. Signal wait time was tau_old = pi/(4*Delta_0) instead of pi/(2*Delta_0).
      2. The control operator had a sign bug for negative phi: the detuning
         term flipped sign along with Omega, but physically only Omega should
         flip when the drive direction reverses.

    Because issue #2 means the old phi values are fundamentally tied to the
    buggy control model, this function extracts the target configuration from
    the old pulse and retrains phi values with the correct physics.

    Parameters
    ----------
    pulse_df_old : DataFrame
        Old-convention pulse schedule with columns "t (us)" and
        "H_x (2pi MHz)" / "H_z (2pi MHz)" (or "Omega_x" / "Omega_z").
    delta_mhz : array-like
        Target detuning centers in nominal MHz.
    alpha_rad : array-like
        Target rotation angles in radians.
    Omega_max_mhz : float
        Maximum Rabi frequency in nominal MHz (default 80).
    Delta_0_mhz : float
        Maximum detuning range in nominal MHz (default 200).
    robustness_window_mhz : float
        Half-width of robustness window in nominal MHz (default 10).
    steps : int
        Number of training steps (default 8000).
    lr : float
        Learning rate (default 5e-2).
    verbose : bool
        Print training progress (default True).
    progress_cb : callable, optional
        Callback(step, total, loss, eta) for progress tracking.

    Returns
    -------
    DataFrame with new-convention columns and correctly trained pulse schedule.
    """
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    # Infer K from old pulse (count control rows = K+1)
    oz_col = "H_z (2pi MHz)" if "H_z (2pi MHz)" in pulse_df_old.columns else "Omega_z (2pi MHz)"
    control_mask = pulse_df_old[oz_col].abs() < 1e-10
    K = control_mask.sum() - 1

    cfg = TrainConfig(
        Omega_max=2 * math.pi * Omega_max_mhz,
        Delta_0=2 * math.pi * Delta_0_mhz,
        robustness_window=2 * math.pi * robustness_window_mhz,
        K=int(K),
        steps=int(steps),
        lr=float(lr),
        device="cpu",
    )

    delta_vals = torch.tensor(np.asarray(delta_mhz, dtype=float)) * (2 * math.pi)
    alpha_vals = torch.tensor(np.asarray(alpha_rad, dtype=float))

    os.makedirs(cfg.out_dir, exist_ok=True)
    phi_final, _, _ = train(
        cfg, delta_vals, alpha_vals,
        sample_size=2048,
        progress_cb=progress_cb,
        verbose=verbose,
        plot_name=os.path.join(cfg.out_dir, "convert_old_to_new.png"),
    )

    # Build new pulse schedule
    tau_us = get_wait_time(cfg.Delta_0)
    t_rows, hx_rows, hz_rows = [], [], []
    for i, pv in enumerate(phi_final.tolist()):
        t_rows.append(np.abs(pv) / cfg.Omega_max)
        hx_rows.append(Omega_max_mhz * np.sign(pv))
        hz_rows.append(0.0)
        if i != len(phi_final) - 1:
            t_rows.append(tau_us)
            hx_rows.append(0.0)
            hz_rows.append(Delta_0_mhz)

    pulse_df_new = pd.DataFrame({
        "t (us)": t_rows,
        "Omega_x (2pi MHz)": hx_rows,
        "Omega_y (2pi MHz)": [0.0] * len(hx_rows),
        "Omega_z (2pi MHz)": hz_rows,
    })

    return pulse_df_new


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    Configuration for QSP training.

    All frequency/detuning fields (Omega_max, Delta_0, robustness_window)
    are in angular units (rad/us). Convert from nominal MHz via:
        value_rad = 2 * pi * value_mhz
    """
    Omega_max: float = 80       # Maximum Rabi frequency (rad/us)
    Delta_0: float = 200         # Maximum detuning range (rad/us); |delta| < Delta_0
    robustness_window: float = 2.0  # Half-width of robustness window (rad/us)
    K: int = 30                 # Number of QSP phases (phi has length K+1)
    steps: int = 8000           # Number of training steps
    lr: float = 5e-2            # Learning rate
    device: str = "cpu"
    out_dir: str = "plots"


def loss_fn(
    phi: torch.Tensor,
    theta_vals: torch.Tensor,
    alpha_vals: torch.Tensor,
    Omega_max: float,
    Delta_0: float,
    lambda_val: float = LAMBDA_VAL,
    # lambda_val: float = 0
):
    """
    QSP training loss: MSE of u_00 element vs target + gradient penalty.

    :param phi: QSP phase angles (K+1,)
    :param theta_vals: sampled QSP angles theta = pi/2 * (1 + delta/Delta_0)
    :param alpha_vals: target rotation angles for each sample
    :param Omega_max: Rabi frequency (angular units, rad/us)
    :param Delta_0: detuning range (angular units, rad/us)
    :param lambda_val: weight for the gradient regularization term
    :return: (loss, pred_u00, target_u00)
    """
    U = build_qsp_unitary(phi, delta=theta_to_delta(theta_vals, Delta_0), Delta_0=Delta_0, Omega=Omega_max)

    pred = U[:, 0, 0]
    target = torch.cos(alpha_vals / 2) - 1j * torch.sin(alpha_vals / 2)

    # Approximate d(pred)/d(theta) via central finite differences
    if lambda_val > 0:
        fd_eps = 1e-2
        U_plus = build_qsp_unitary(phi, delta=theta_to_delta(theta_vals + fd_eps, Delta_0), Delta_0=Delta_0, Omega=Omega_max)
        U_minus = build_qsp_unitary(phi, delta=theta_to_delta(theta_vals - fd_eps, Delta_0), Delta_0=Delta_0, Omega=Omega_max)
        grad_pred = (U_plus[:, 0, 0] - U_minus[:, 0, 0]) / (2 * fd_eps)

    target_err = pred - target
    if lambda_val > 0:
        loss = (target_err.abs() ** 2 + lambda_val * grad_pred.abs() ** 2).mean()
    else:
        loss = (target_err.abs() ** 2).mean()
    return loss, pred, target


def train_epoch(
    phase_model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: Optional[torch.optim.lr_scheduler._LRScheduler],
    cfg: TrainConfig,
    theta_samples: torch.Tensor,
    alpha_samples: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Perform one training step."""
    phi = phase_model()
    loss, pred, target = loss_fn(
        phi, theta_samples, alpha_samples, cfg.Omega_max, cfg.Delta_0)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(phase_model.parameters(), max_norm=10.0)
    opt.step()
    if sched is not None:
        sched.step()
    return loss.item(), pred, target


def train(
    cfg: TrainConfig,
    delta_vals: torch.Tensor,
    alpha_vals: torch.Tensor,
    sample_size: int = 1024,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
    verbose: bool = True,
    plot_name: str = None
):
    assert delta_vals.shape == alpha_vals.shape, "delta_vals and alpha_vals must have the same shape."
    delta_vals = delta_vals.to(cfg.device)
    alpha_vals = alpha_vals.to(cfg.device)

    # Sample around provided delta_vals within +/- robustness_window
    idx = torch.randint(0, delta_vals.numel(), (sample_size,), device=cfg.device)
    centers = delta_vals[idx]
    half_window = cfg.robustness_window
    jitter = (2.0 * torch.rand(sample_size, device=cfg.device) - 1.0) * half_window
    delta_samples = (centers + jitter).clamp(-cfg.Delta_0, cfg.Delta_0)
    theta_samples = delta_to_theta(delta_samples, cfg.Delta_0)
    nearest_idx = (delta_samples[:, None] - delta_vals[None, :]).abs().argmin(dim=1)
    alpha_samples = alpha_vals[nearest_idx]

    phase_model = DirectPhases(cfg.K).to(cfg.device)
    opt = torch.optim.Adam(phase_model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    best_phi = None
    best_loss = float("inf")

    start_t = time.perf_counter()
    if verbose:
        with tqdm(total=cfg.steps, desc="Training", dynamic_ncols=True) as pbar:
            for step in range(1, cfg.steps + 1):
                loss, pred, target = train_epoch(
                    phase_model, opt, sched, cfg,
                    theta_samples, alpha_samples
                )
                if loss < best_loss:
                    best_loss = loss
                    best_phi = phase_model().detach().cpu().clone()
                pbar.set_postfix({"step": step, "train_loss": f"{loss:.3e}"})
                pbar.update(1)
                if progress_cb is not None:
                    elapsed = time.perf_counter() - start_t
                    rate = elapsed / max(step, 1)
                    eta = rate * (cfg.steps - step)
                    progress_cb(step, cfg.steps, loss, eta)
    else:
        for step in range(1, cfg.steps + 1):
            loss, pred, target = train_epoch(
                phase_model, opt, sched, cfg,
                theta_samples, alpha_samples
            )
            if loss < best_loss:
                best_loss = loss
                best_phi = phase_model().detach().cpu().clone()

    if plot_name is None:
        plot_name = os.path.join(cfg.out_dir, f"matrix_element_vs_delta_K{cfg.K}_loss_{best_loss:.6f}.png")

    best_fidelity = plot_matrix_element_vs_delta(
        best_phi, cfg, delta_vals, alpha_vals, out_path=plot_name,
    )
    return best_phi, best_loss, best_fidelity


# ─────────────────────────────────────────────────────────────────────────────
#  Runtime
# ─────────────────────────────────────────────────────────────────────────────

def get_control_runtime(phi, cfg):
    """Total QSP sequence runtime in microseconds."""
    tau = get_wait_time(cfg.Delta_0)
    K = len(phi) - 1
    T = K * tau + sum(phi.abs()).item() / cfg.Omega_max
    return T


# ─────────────────────────────────────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_matrix_element_vs_delta(
    phi: torch.Tensor,
    cfg: TrainConfig,
    delta_vals: torch.Tensor,
    alpha_vals: torch.Tensor,
    out_path: str
):
    """Plot U00 and U01 vs detuning with target windows overlaid."""
    delta_range = torch.linspace(-cfg.Delta_0, cfg.Delta_0, steps=1024, device=cfg.device)

    U = build_qsp_unitary(phi, delta_range, cfg.Delta_0, cfg.Omega_max)

    Hadamard = 1 / math.sqrt(2) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128, device=cfg.device)
    U = Hadamard @ U @ Hadamard  # QSP basis -> physical basis

    u00 = U[:, 0, 0].detach().cpu()
    u01 = U[:, 0, 1].detach().cpu()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert angular (rad/us) to nominal (MHz) for display
    ax.plot(delta_range / (2 * math.pi), u00.real, label="Re(u_00)")
    ax.plot(delta_range / (2 * math.pi), u01.imag, label="Im(u_01)")

    for i, delta in enumerate(delta_vals):
        alpha_i = alpha_vals[i]
        delta_min = delta.item() - cfg.robustness_window
        delta_max = delta.item() + cfg.robustness_window

        ax.hlines(
            y=np.cos(alpha_i.item() / 2),
            xmin=delta_min / (2 * math.pi),
            xmax=delta_max / (2 * math.pi),
            colors="red", linestyles="dashed",
            label="Re(target u_00)" if i == 0 else None
        )
        ax.hlines(
            y=-np.sin(alpha_i.item() / 2),
            xmin=delta_min / (2 * math.pi),
            xmax=delta_max / (2 * math.pi),
            colors="green", linestyles="dashed",
            label="Im(target u_01)" if i == 0 else None
        )

        ax.axvspan(delta_min / (2 * math.pi), delta_max / (2 * math.pi), color="gray", alpha=0.15)
        label = f"R_x({alpha_i / math.pi:.4f} pi)"
        ax.text(delta.item() / (2 * math.pi), 1.05, label, ha="center", va="bottom", fontsize=9)

    T = get_control_runtime(phi, cfg)
    fidelity_val = fidelity(phi, delta_vals, alpha_vals, cfg)

    ax.set_xlabel("Detuning δ (MHz)")
    ax.set_ylabel("Matrix Element")
    ax.set_ylim(-1.2, 1.2)
    title_str = (
        "Matrix Element vs Target Controlled Rx(alpha)\n"
        f"Rabi: (2pi) {cfg.Omega_max / (2 * math.pi):.2f} MHz\n"
        f"Total Time T={T:.6f} μs\nFidelity: {fidelity_val:.6f}"
    )
    ax.set_title(title_str)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return fidelity_val


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=30, help="Max phase index K (phi has length K+1).")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--num_peaks", type=int, default=4, help="Number of peaks in target function P(x).")
    ap.add_argument("--Delta_0", type=float, default=40.0, help="Maximum detuning width (nominal MHz)")
    ap.add_argument("--robustness_window", type=float, default=2.0, help="Half-width of robustness window (nominal MHz)")
    ap.add_argument("--Omega_max", type=float, default=80.0, help="Maximum Rabi frequency (nominal MHz)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out_dir", type=str, default="plots_relaxed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Convert nominal MHz -> angular rad/us
    cfg = TrainConfig(
        Omega_max=2 * math.pi * args.Omega_max,
        Delta_0=2 * math.pi * args.Delta_0,
        robustness_window=2 * math.pi * args.robustness_window,
        K=args.K,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        out_dir=args.out_dir,
    )

    device = torch.device(cfg.device)
    torch.set_default_dtype(torch.float64)

    K = args.K

    delta_vals = torch.linspace(-cfg.Delta_0, cfg.Delta_0, steps=args.num_peaks, device=device)
    alpha_vals = torch.tensor([0.0, math.pi / 2, math.pi, math.pi / 3], device=device)

    phi_final, final_loss, _ = train(cfg, delta_vals, alpha_vals, sample_size=2048)

    phi_df = pd.DataFrame({"index": np.arange(len(phi_final)), "phi": phi_final.numpy()})
    phi_df.to_csv(os.path.join(cfg.out_dir, f"learned_phases_K={K}_final_loss_{final_loss:.6f}.csv"), index=True)


if __name__ == "__main__":
    main()
