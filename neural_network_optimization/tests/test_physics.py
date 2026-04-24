import math

import numpy as np
import pandas as pd
import torch

from neural_network_optimization.physics import (
    Rz,
    signal_operator,
    bmm,
    delta_to_theta,
    theta_to_delta,
    build_qsp_unitary,
    build_qsp_unitary_batched,
    fidelity_from_pulse,
)


def _unitarity_error(U: torch.Tensor) -> float:
    I = torch.eye(2, dtype=U.dtype, device=U.device).expand_as(U)
    return float((U @ U.conj().transpose(-1, -2) - I).abs().max().item())


def test_signal_operator_unitary_and_limits():
    theta = torch.tensor([0.0, math.pi / 3, math.pi, 2 * math.pi], dtype=torch.float64)
    U = signal_operator(theta)
    assert U.shape == (4, 2, 2)
    assert _unitarity_error(U) < 1e-12

    # theta=0 -> identity
    I = torch.eye(2, dtype=torch.complex128)
    assert (U[0] - I).abs().max().item() < 1e-12
    # theta=pi -> -i sigma_x
    sx = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    assert (U[2] - (-1j * sx)).abs().max().item() < 1e-12


def test_Rz_matches_closed_form():
    phi = torch.tensor([0.0, math.pi / 2, -math.pi], dtype=torch.float64)
    U = Rz(phi)
    expected_00 = torch.exp(-0.5j * phi.to(torch.complex128))
    expected_11 = torch.exp(+0.5j * phi.to(torch.complex128))
    assert (U[:, 0, 0] - expected_00).abs().max().item() < 1e-12
    assert (U[:, 1, 1] - expected_11).abs().max().item() < 1e-12
    assert U[:, 0, 1].abs().max().item() < 1e-14
    assert _unitarity_error(U) < 1e-12


def test_delta_theta_roundtrip():
    Delta_0 = 2 * math.pi * 200.0
    theta = torch.linspace(0, math.pi, 17, dtype=torch.float64)
    rt = delta_to_theta(theta_to_delta(theta, Delta_0), Delta_0)
    assert (rt - theta).abs().max().item() < 1e-12


def test_bmm_shape():
    A = torch.randn(5, 2, 2, dtype=torch.complex128)
    B = torch.randn(5, 2, 2, dtype=torch.complex128)
    C = bmm(A, B)
    assert C.shape == (5, 2, 2)
    assert (C - A @ B).abs().max().item() < 1e-12


def test_build_qsp_identity_at_zero_phi():
    # All-zero phi sequence with Omega>0 gives identity control; then K signal
    # operators stack up as signal(theta)^K which is NOT identity -- so test
    # K = 0 (no signals) for identity.
    Omega = 2 * math.pi * 80.0
    Delta_0 = 2 * math.pi * 200.0
    phi = torch.zeros(1, dtype=torch.float64)  # K = 0, just one control
    delta = torch.tensor([0.0, 50.0, -50.0], dtype=torch.float64) * (2 * math.pi)
    U = build_qsp_unitary(phi, delta, Delta_0, Omega)
    I = torch.eye(2, dtype=torch.complex128).expand(3, 2, 2)
    assert (U - I).abs().max().item() < 1e-12


def test_build_qsp_unitary_for_random_phi():
    torch.manual_seed(0)
    Omega = 2 * math.pi * 40.0
    Delta_0 = 2 * math.pi * 200.0
    K = 8
    phi = math.pi * torch.randn(K + 1, dtype=torch.float64)
    delta = (2 * math.pi) * torch.tensor([-100.0, -32.0, 32.0, 100.0], dtype=torch.float64)
    U = build_qsp_unitary(phi, delta, Delta_0, Omega)
    assert U.shape == (4, 2, 2)
    assert _unitarity_error(U) < 1e-10


def test_batched_matches_unbatched():
    torch.manual_seed(1)
    Omega = 2 * math.pi * 40.0
    Delta_0 = 2 * math.pi * 200.0
    K = 10
    B = 3
    phi = math.pi * torch.randn(B, K + 1, dtype=torch.float64)
    delta = (2 * math.pi) * torch.linspace(-200, 200, 7, dtype=torch.float64)

    U_re, U_im = build_qsp_unitary_batched(phi, delta, Delta_0, Omega)
    U_batched = U_re + 1j * U_im  # (B, D, 2, 2)

    for b in range(B):
        U_single = build_qsp_unitary(phi[b], delta, Delta_0, Omega)  # (D, 2, 2)
        diff = (U_batched[b] - U_single).abs().max().item()
        assert diff < 1e-10, f"batched mismatch at b={b}: {diff}"


def test_fidelity_empty_pulse_identity():
    # A single zero-duration row should yield identity => perfect fidelity for alpha=0.
    pulse_df = pd.DataFrame({
        "t (us)": [0.0],
        "Omega_x (2pi MHz)": [0.0],
        "Omega_y (2pi MHz)": [0.0],
        "Omega_z (2pi MHz)": [0.0],
    })
    F = fidelity_from_pulse(
        pulse_df, delta_mhz=np.array([0.0]), alpha_rad=np.array([0.0]),
        robustness_window_mhz=1.0, sample_size=32,
    )
    assert F > 1 - 1e-10
