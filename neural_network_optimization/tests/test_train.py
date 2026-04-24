import os

import torch

from neural_network_optimization.train import train, _compute_loss, presample_detunings
from neural_network_optimization.model import PulseNet, get_weight_path


def test_train_runs_and_reduces_loss(tmp_path):
    weight_dir = str(tmp_path / "weights")
    torch.manual_seed(0)

    # Record initial eval loss separately (for comparison) — rebuild same config.
    Omega, K = 80, 5
    net0 = PulseNet(Omega=Omega, K=K)
    delta_all, peak_ids = presample_detunings(
        net0.delta_centers_ang, net0.robustness_window_ang, net0.Delta_0_ang,
        samples_per_peak=16, device=None,
    )
    from neural_network_optimization.data import build_dataset
    _, alpha_eval = build_dataset(n_train=256, n_eval=64, seed=0)
    with torch.no_grad():
        initial_loss = _compute_loss(net0, alpha_eval[:32], delta_all, peak_ids).item()

    # Train a small config
    net = train(
        Omega=Omega, K=K,
        n_train=256, n_eval=64, batch_size=32, epochs=4,
        samples_per_peak=16, weight_dir=weight_dir, seed=0, verbose=False,
    )
    path = get_weight_path(weight_dir, Omega, K)
    assert os.path.exists(path), f"Expected weights at {path}"

    with torch.no_grad():
        final_loss = _compute_loss(net, alpha_eval[:32], delta_all, peak_ids).item()

    assert final_loss < initial_loss, (
        f"Training did not reduce loss: initial={initial_loss:.4e} final={final_loss:.4e}"
    )
