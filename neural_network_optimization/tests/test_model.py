import math

import torch

from neural_network_optimization.model import (
    PulseNet,
    phi_to_pulse_df,
    get_runtime_from_phi,
)


def test_forward_shape_and_dtype():
    net = PulseNet(Omega=80, K=5)
    alpha = torch.rand(7, 4, dtype=torch.float64) * 2 * math.pi
    phi = net(alpha)
    assert phi.shape == (7, 6)
    assert phi.dtype == torch.float64


def test_1d_input_returns_1d_output():
    net = PulseNet(Omega=80, K=5)
    alpha = torch.zeros(4, dtype=torch.float64)
    phi = net(alpha)
    assert phi.shape == (6,)


def test_phi_near_zero_at_init():
    torch.manual_seed(0)
    net = PulseNet(Omega=80, K=10)
    alpha = torch.rand(4, 4, dtype=torch.float64) * 2 * math.pi
    phi = net(alpha)
    # The ×0.01 output-layer init scales outputs to small values.
    assert phi.abs().max().item() < 0.1


def test_4pi_periodicity():
    torch.manual_seed(1)
    net = PulseNet(Omega=80, K=12)
    alpha = torch.rand(5, 4, dtype=torch.float64) * 2 * math.pi
    phi_a = net(alpha)
    phi_b = net(alpha + 4 * math.pi)
    assert (phi_a - phi_b).abs().max().item() < 1e-10


def test_phi_to_pulse_df_columns_and_length():
    K = 5
    phi = torch.tensor([0.3, -0.4, 0.5, 0.0, 0.7, -0.1], dtype=torch.float64)
    df = phi_to_pulse_df(phi, Omega_mhz=80, Delta_0_mhz=200)
    expected_cols = {"t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"}
    assert set(df.columns) == expected_cols
    assert len(df) == 2 * K + 1  # K+1 controls + K signal waits
    assert (df["t (us)"] >= 0).all()


def test_get_runtime_formula():
    phi = torch.tensor([0.5, -0.5, 0.5], dtype=torch.float64)
    Omega_ang = 2 * math.pi * 80
    Delta_0_ang = 2 * math.pi * 200
    K = phi.numel() - 1  # 2
    expected = K * math.pi / (2 * Delta_0_ang) + float(phi.abs().sum().item()) / Omega_ang
    got = get_runtime_from_phi(phi, Omega_ang, Delta_0_ang)
    assert abs(got - expected) < 1e-12
