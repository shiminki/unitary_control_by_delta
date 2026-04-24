"""QSP physics primitives.

Conventions:

    QSP basis: signal = R_x(theta), control = Z-drive + X-detuning.
    U = control(phi_0; delta) signal(theta) ... control(phi_K; delta)

Physical-basis u_00 (= u_00 of R_x(alpha) on the qubit) is recovered at the
viewer via a Hadamard sandwich — see plotting.plot_matrix_element.  For training
and fidelity targets we work directly in the QSP basis where the target is
R_z(alpha) whose u_00 equals exp(-i*alpha/2).
"""

import math

import numpy as np
import pandas as pd
import torch


# ─────────────────────────────────────────────────────────────────────────────
#  Matrix primitives
# ─────────────────────────────────────────────────────────────────────────────

def Rz(phi: torch.Tensor) -> torch.Tensor:
    """Rz(phi) = diag(e^{-i phi/2}, e^{+i phi/2}).  phi: (B,) real."""
    phi = phi.to(dtype=torch.float64)
    e0 = torch.exp(-0.5j * phi)
    e1 = torch.exp(+0.5j * phi)
    B = phi.shape[0] if phi.ndim > 0 else 1
    out = torch.zeros((B, 2, 2), dtype=torch.complex128, device=phi.device)
    out[:, 0, 0] = e0.reshape(-1)
    out[:, 1, 1] = e1.reshape(-1)
    return out


def signal_operator(theta: torch.Tensor) -> torch.Tensor:
    """QSP signal operator R_x(theta) = exp(-i theta/2 sigma_x).  theta: (B,)."""
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
    return torch.bmm(A, B)


# ─────────────────────────────────────────────────────────────────────────────
#  Delta <-> theta mapping
# ─────────────────────────────────────────────────────────────────────────────

def delta_to_theta(delta: torch.Tensor, Delta_0: float) -> torch.Tensor:
    """theta = pi/2 * (1 + delta/Delta_0)."""
    return math.pi / 2 * (1 + delta / Delta_0)


def theta_to_delta(theta: torch.Tensor, Delta_0: float) -> torch.Tensor:
    """Inverse of delta_to_theta."""
    return (2 / math.pi * theta - 1) * Delta_0


def get_wait_time(Delta_0_ang: float) -> float:
    """Signal wait time (us), with Delta_0 in angular units (rad/us)."""
    return math.pi / (2 * Delta_0_ang)


# ─────────────────────────────────────────────────────────────────────────────
#  QSP unitary builders
# ─────────────────────────────────────────────────────────────────────────────

def build_qsp_unitary(
    phi: torch.Tensor,
    delta: torch.Tensor,
    Delta_0: float,
    Omega: float,
) -> torch.Tensor:
    """Build the noisy QSP unitary for a batch of detuning values.

    Sequence: control(phi_0; delta) signal(theta) ... control(phi_K; delta).

    Control Hamiltonian: H_c = sign(phi) * Omega/2 * sigma_z + delta/2 * sigma_x,
    integrated for t = |phi| / Omega.

    Parameters
    ----------
    phi     : (K+1,) real phase angles (dimensionless).
    delta   : (B,)   detuning values (angular units, rad/us).
    Delta_0 : float  maximum detuning range (angular units).
    Omega   : float  Rabi frequency (angular units).

    Returns
    -------
    (B, 2, 2) complex128 unitary.
    """
    assert phi.ndim == 1, "phi must be shape [K+1]"
    theta = delta_to_theta(delta, Delta_0)
    K = phi.shape[0] - 1
    B = theta.shape[0]
    dev = theta.device

    def apply_control(Ucur, phase, delta):
        Omega_t = torch.full((B,), Omega, dtype=torch.float64, device=dev)
        norm = torch.sqrt(Omega_t ** 2 + delta ** 2)
        abs_lamb = torch.abs(phase) / (2 * Omega) * norm
        sign_phase = torch.sign(phase)
        c = torch.cos(abs_lamb)
        s = torch.sin(abs_lamb)
        sin_diag = s * sign_phase * Omega_t / norm
        sin_offdiag = s * delta / norm

        R = torch.zeros((B, 2, 2), dtype=torch.complex128, device=dev)
        R[:, 0, 0] = c - 1j * sin_diag
        R[:, 1, 1] = c + 1j * sin_diag
        R[:, 0, 1] = -1j * sin_offdiag
        R[:, 1, 0] = -1j * sin_offdiag
        return bmm(R, Ucur)

    I = torch.eye(2, dtype=torch.complex128, device=dev).expand(B, 2, 2).clone()
    W = signal_operator(theta)

    U = apply_control(I, phi[0], delta)
    for j in range(K):
        U = bmm(W, U)
        U = apply_control(U, phi[j + 1], delta)
    return U


def build_qsp_unitary_batched(
    phi: torch.Tensor,
    delta: torch.Tensor,
    Delta_0: float,
    Omega: float,
):
    """Vectorized / all-real QSP builder.

    Parameters
    ----------
    phi     : (B, K+1) real phase angles.
    delta   : (D,)     detuning values (angular units).
    Delta_0 : float    max detuning (angular).
    Omega   : float    Rabi (angular).

    Returns
    -------
    (U_re, U_im) both (B, D, 2, 2) float64.  U = U_re + i*U_im.
    """
    assert phi.ndim == 2, "phi must be (B, K+1)"
    B, Kp1 = phi.shape
    K = Kp1 - 1
    D = delta.shape[0]
    dev = delta.device
    BD = B * D

    theta = delta_to_theta(delta, Delta_0)

    delta_flat = delta.unsqueeze(0).expand(B, D).reshape(BD)
    theta_flat = theta.unsqueeze(0).expand(B, D).reshape(BD)

    wc = torch.cos(theta_flat / 2)
    ws = torch.sin(theta_flat / 2)

    norm = torch.sqrt(Omega ** 2 + delta_flat ** 2)

    u00_re = torch.ones(BD, dtype=torch.float64, device=dev)
    u00_im = torch.zeros(BD, dtype=torch.float64, device=dev)
    u01_re = torch.zeros(BD, dtype=torch.float64, device=dev)
    u01_im = torch.zeros(BD, dtype=torch.float64, device=dev)
    u10_re = torch.zeros(BD, dtype=torch.float64, device=dev)
    u10_im = torch.zeros(BD, dtype=torch.float64, device=dev)
    u11_re = torch.ones(BD, dtype=torch.float64, device=dev)
    u11_im = torch.zeros(BD, dtype=torch.float64, device=dev)

    for j in range(K + 1):
        phase_flat = phi[:, j].unsqueeze(1).expand(B, D).reshape(BD)
        abs_lamb = torch.abs(phase_flat) / (2 * Omega) * norm
        sign_phase = torch.sign(phase_flat)
        c = torch.cos(abs_lamb)
        s = torch.sin(abs_lamb)
        sd = s * sign_phase * Omega / norm
        so = s * delta_flat / norm

        n00_re = c * u00_re + sd * u00_im + so * u10_im
        n00_im = c * u00_im - sd * u00_re - so * u10_re
        n01_re = c * u01_re + sd * u01_im + so * u11_im
        n01_im = c * u01_im - sd * u01_re - so * u11_re
        n10_re = so * u00_im + c * u10_re - sd * u10_im
        n10_im = -so * u00_re + c * u10_im + sd * u10_re
        n11_re = so * u01_im + c * u11_re - sd * u11_im
        n11_im = -so * u01_re + c * u11_im + sd * u11_re
        u00_re, u00_im = n00_re, n00_im
        u01_re, u01_im = n01_re, n01_im
        u10_re, u10_im = n10_re, n10_im
        u11_re, u11_im = n11_re, n11_im

        if j < K:
            n00_re = wc * u00_re + ws * u10_im
            n00_im = wc * u00_im - ws * u10_re
            n01_re = wc * u01_re + ws * u11_im
            n01_im = wc * u01_im - ws * u11_re
            n10_re = ws * u00_im + wc * u10_re
            n10_im = -ws * u00_re + wc * u10_im
            n11_re = ws * u01_im + wc * u11_re
            n11_im = -ws * u01_re + wc * u11_im
            u00_re, u00_im = n00_re, n00_im
            u01_re, u01_im = n01_re, n01_im
            u10_re, u10_im = n10_re, n10_im
            u11_re, u11_im = n11_re, n11_im

    U_re = torch.stack([
        torch.stack([u00_re, u01_re], dim=-1),
        torch.stack([u10_re, u11_re], dim=-1),
    ], dim=-2).reshape(B, D, 2, 2)
    U_im = torch.stack([
        torch.stack([u00_im, u01_im], dim=-1),
        torch.stack([u10_im, u11_im], dim=-1),
    ], dim=-2).reshape(B, D, 2, 2)
    return U_re, U_im


# ─────────────────────────────────────────────────────────────────────────────
#  Pulse-schedule fidelity (physical basis, evaluator only)
# ─────────────────────────────────────────────────────────────────────────────

def fidelity_from_pulse(
    pulse_df: pd.DataFrame,
    delta_mhz,
    alpha_rad,
    robustness_window_mhz: float,
    sample_size: int = 5000,
) -> float:
    """Evaluate average gate fidelity from a pulse schedule DataFrame.

    Propagates U step-by-step in the physical basis (H = (Omega_x/2) sigma_x +
    ((Omega_z + delta)/2) sigma_z) under random detuning samples within
    +/- robustness_window_mhz of each target delta.  Target at peak i is
    R_x(alpha_i).  F = (|Tr(U_target^dag U)|^2 + 2) / 6, averaged.
    """
    ts = pulse_df["t (us)"].to_numpy(dtype=float)
    if "Omega_x (2pi MHz)" in pulse_df.columns:
        Omega_xs = pulse_df["Omega_x (2pi MHz)"].to_numpy(dtype=float)
        Omega_zs = pulse_df["Omega_z (2pi MHz)"].to_numpy(dtype=float)
    else:
        Omega_xs = pulse_df["H_x (2pi MHz)"].to_numpy(dtype=float)
        Omega_zs = pulse_df["H_z (2pi MHz)"].to_numpy(dtype=float)

    delta_arr = (
        delta_mhz.detach().cpu().numpy().astype(float)
        if isinstance(delta_mhz, torch.Tensor)
        else np.asarray(delta_mhz, dtype=float)
    )
    alpha_arr = (
        alpha_rad.detach().cpu().numpy().astype(float)
        if isinstance(alpha_rad, torch.Tensor)
        else np.asarray(alpha_rad, dtype=float)
    )

    rng = np.random.default_rng()
    all_fid = []

    for delta_i, alpha_i in zip(delta_arr, alpha_arr):
        delta_s = rng.uniform(
            delta_i - robustness_window_mhz,
            delta_i + robustness_window_mhz,
            size=sample_size,
        )
        c_tgt = math.cos(alpha_i / 2)
        s_tgt = math.sin(alpha_i / 2)

        u00 = np.ones(sample_size, dtype=complex)
        u01 = np.zeros(sample_size, dtype=complex)
        u10 = np.zeros(sample_size, dtype=complex)
        u11 = np.ones(sample_size, dtype=complex)

        for j in range(len(ts)):
            a = math.pi * Omega_xs[j] * ts[j]
            b = math.pi * (Omega_zs[j] + delta_s) * ts[j]
            r = np.sqrt(a ** 2 + b ** 2)
            cos_r = np.cos(r)
            with np.errstate(invalid="ignore", divide="ignore"):
                sinc_r = np.where(r > 1e-15, np.sin(r) / r, 1.0)

            v00 = cos_r - 1j * b * sinc_r
            v01 = -1j * a * sinc_r
            v11 = cos_r + 1j * b * sinc_r

            new_u00 = v00 * u00 + v01 * u10
            new_u01 = v00 * u01 + v01 * u11
            new_u10 = v01 * u00 + v11 * u10
            new_u11 = v01 * u01 + v11 * u11
            u00, u01, u10, u11 = new_u00, new_u01, new_u10, new_u11

        traces = (
            np.conj(u00) * c_tgt
            + np.conj(u10) * (-1j * s_tgt)
            + np.conj(u01) * (-1j * s_tgt)
            + np.conj(u11) * c_tgt
        )
        all_fid.append((np.abs(traces) ** 2 + 2.0) / 6.0)

    return float(np.concatenate(all_fid).mean())
