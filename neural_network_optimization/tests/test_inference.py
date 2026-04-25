import math

import torch

from neural_network_optimization.inference import (
    load_model,
    generate_phi,
    generate_pulse,
    compute_runtime,
)
from neural_network_optimization.model import PulseNet, get_weight_path, get_runtime_from_phi
from neural_network_optimization.train import train


def test_load_model_roundtrip(tmp_path):
    weight_dir = str(tmp_path / "weights")
    Omega, K = 80, 5
    torch.manual_seed(0)

    trained = train(
        Omega=Omega, K=K,
        n_train=128, n_eval=32, batch_size=32, epochs=2,
        samples_per_peak=8, weight_dir=weight_dir, seed=0, verbose=False,
    )
    loaded = load_model(Omega=Omega, K=K, weight_dir=weight_dir)

    alpha = [0.4, 1.2, -0.6, 2.7]
    phi_trained = generate_phi(trained, alpha)
    phi_loaded = generate_phi(loaded, alpha)
    assert (phi_trained - phi_loaded).abs().max().item() < 1e-12


def test_pulse_df_columns(tmp_path):
    weight_dir = str(tmp_path / "weights")
    net = train(
        Omega=80, K=5, n_train=128, n_eval=32, batch_size=32, epochs=1,
        samples_per_peak=8, weight_dir=weight_dir, seed=0, verbose=False,
    )
    df = generate_pulse(net, [0.0, 0.0, 0.0, 0.0])
    expected = {"t (us)", "Omega_x (2pi MHz)", "Omega_y (2pi MHz)", "Omega_z (2pi MHz)"}
    assert set(df.columns) == expected
    assert (df["t (us)"] >= 0).all()


def test_compute_runtime_matches_formula():
    net = PulseNet(Omega=80, K=7)
    alpha = [0.0, 0.0, 0.0, 0.0]
    phi = generate_phi(net, alpha)
    expected = get_runtime_from_phi(phi, net.Omega_ang, net.Delta_0_ang)
    got = compute_runtime(net, alpha)
    assert abs(got - expected) < 1e-12
    # With tiny phi (near-zero), runtime is close to the pure signal-wait baseline.
    baseline = net.K * math.pi / (2 * net.Delta_0_ang)
    assert got >= baseline - 1e-12
