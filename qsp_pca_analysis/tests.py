"""Tests for the QSP PCA analysis pipeline."""

import math
import os
import shutil
import unittest

import numpy as np
import torch

from single_pulse_optimization_QSP.qsp_fit_x_rotation import fidelity_from_pulse

from .constants import DEFAULT_K, DELTA_0_MHZ, OMEGA_MHZ
from .model import PulseGeneratorNet
from .pulse import phi_to_pulse_df
from .training import presample_detunings, compute_batch_loss
from .pca import pca_analysis


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
