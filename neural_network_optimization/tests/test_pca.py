import math

import numpy as np
import torch

from neural_network_optimization.model import PulseNet
from neural_network_optimization.pca_analysis import (
    run_pca,
    reconstruct_phi_analytical,
    reconstruct_phi_polyfit,
)


def test_run_pca_shape_and_keys():
    torch.manual_seed(0)
    net = PulseNet(Omega=80, K=5)
    res = run_pca(net, peak_index=0, n_samples=64, save_figures=False)
    K_plus_1 = net.K + 1

    assert res["alpha_grid"].shape == (64,)
    assert res["phi_matrix"].shape == (64, K_plus_1)
    assert res["mean_phi"].shape == (K_plus_1,)
    assert res["principal_components"].shape[0] == K_plus_1
    assert res["n_effective_dof"] >= 1
    assert res["amplitudes"].shape == (64, res["n_effective_dof"])
    assert len(res["amplitude_fits"]) == res["n_effective_dof"]
    assert len(res["polyfit_fits"]) == res["n_effective_dof"]
    # Variance bookkeeping
    total = res["explained_variance_ratio"].sum()
    assert abs(total - 1.0) < 1e-9


def test_reconstruction_returns_correct_shape():
    torch.manual_seed(1)
    net = PulseNet(Omega=40, K=7)
    res = run_pca(net, peak_index=1, n_samples=48, save_figures=False)
    K_plus_1 = net.K + 1

    phi_fourier = reconstruct_phi_analytical(res, alpha=1.2)
    phi_poly = reconstruct_phi_polyfit(res, alpha=1.2)
    assert phi_fourier.shape == (K_plus_1,)
    assert phi_poly.shape == (K_plus_1,)

    # With n_basis=0 the reconstruction should equal the mean phi
    mean = res["mean_phi"]
    phi0 = reconstruct_phi_analytical(res, alpha=3.14, n_basis=0)
    assert np.allclose(phi0, mean, atol=1e-12)


def test_pca_variance_monotone():
    torch.manual_seed(2)
    net = PulseNet(Omega=80, K=9)
    res = run_pca(net, peak_index=2, n_samples=80, save_figures=False)
    er = res["explained_variance_ratio"]
    # Non-increasing
    assert all(er[i] >= er[i + 1] - 1e-12 for i in range(len(er) - 1))
