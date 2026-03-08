"""
Test suite for QSP-based detuning-controlled unitary.

Usage:
    python test.py --f implementation     # unit tests for correctness
    python test.py --f scaling_small      # verify scaling_law_small CSV results
    python test.py --f scaling_large      # verify scaling_law_results CSV results
    python test.py --f scaling            # verify both scaling datasets
    python test.py --f all                # run all tests
"""

import argparse
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
#  Imports from the project
# ---------------------------------------------------------------------------
from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    TrainConfig,
    build_qsp_unitary,
    delta_to_theta,
    theta_to_delta,
    get_control_runtime,
    fidelity as qsp_fidelity,
    train,
    signal_operator,
    Rz,
    bmm,
)

# ---------------------------------------------------------------------------
#  Constants used across tests
# ---------------------------------------------------------------------------
DELTA_0_MHZ = 200.0
OMEGA_MAX_MHZ = 80.0
SIGNAL_WINDOW_MHZ = 10.0

DELTA_0 = 2 * math.pi * DELTA_0_MHZ       # angular (rad/us)
OMEGA_MAX = 2 * math.pi * OMEGA_MAX_MHZ
SIGNAL_WINDOW = 2 * math.pi * SIGNAL_WINDOW_MHZ


# ═══════════════════════════════════════════════════════════════════════════
#  IMPLEMENTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDeltaThetaMapping(unittest.TestCase):
    """Verify delta <-> theta conversion round-trips and boundary values."""

    def test_roundtrip(self):
        deltas = torch.linspace(-DELTA_0, DELTA_0, 200, dtype=torch.float64)
        recovered = theta_to_delta(delta_to_theta(deltas, DELTA_0), DELTA_0)
        self.assertTrue(
            torch.allclose(deltas, recovered, atol=1e-10),
            "delta -> theta -> delta round-trip failed",
        )

    def test_boundary_values(self):
        """delta=-Delta_0 => theta=0, delta=0 => pi/2, delta=+Delta_0 => pi."""
        cases = [
            (-DELTA_0, 0.0),
            (0.0, math.pi / 2),
            (DELTA_0, math.pi),
        ]
        for delta, expected_theta in cases:
            theta = delta_to_theta(torch.tensor(delta, dtype=torch.float64), DELTA_0)
            self.assertAlmostEqual(
                theta.item(), expected_theta, places=10,
                msg=f"delta={delta:.4f} => theta should be {expected_theta}",
            )

    def test_cos_half_theta_range(self):
        """cos(theta/2) should map [-Delta_0, Delta_0] -> [0, 1]."""
        deltas = torch.linspace(-DELTA_0, DELTA_0, 500, dtype=torch.float64)
        thetas = delta_to_theta(deltas, DELTA_0)
        cos_vals = torch.cos(thetas / 2)
        self.assertGreaterEqual(cos_vals.min().item(), -1e-12)
        self.assertLessEqual(cos_vals.max().item(), 1.0 + 1e-12)
        # At delta = -Delta_0: cos(0) = 1
        self.assertAlmostEqual(cos_vals[0].item(), 1.0, places=10)
        # At delta = +Delta_0: cos(pi/2) = 0
        self.assertAlmostEqual(cos_vals[-1].item(), 0.0, places=10)


class TestSignalOperator(unittest.TestCase):
    """Verify signal_operator(theta) = R_x(theta) = exp(-i theta/2 sigma_x)."""

    def test_signal_operator_is_Rx(self):
        """signal_operator(theta) matrix elements should match R_x(theta)."""
        for angle in [0.0, math.pi / 6, math.pi / 3, math.pi / 2, math.pi]:
            theta = torch.tensor([angle], dtype=torch.float64)
            Wt = signal_operator(theta)[0]  # (2, 2)
            c = math.cos(angle / 2)
            s = math.sin(angle / 2)
            expected = torch.tensor(
                [[c, -1j * s], [-1j * s, c]], dtype=torch.complex128
            )
            self.assertTrue(
                torch.allclose(Wt, expected, atol=1e-12),
                f"signal_operator({angle}) != R_x({angle})",
            )

    def test_signal_operator_is_unitary(self):
        thetas = torch.linspace(0, math.pi, 50, dtype=torch.float64)
        Ws = signal_operator(thetas)  # (50, 2, 2)
        I2 = torch.eye(2, dtype=torch.complex128).unsqueeze(0).expand(50, -1, -1)
        product = torch.bmm(Ws, Ws.conj().transpose(-1, -2))
        self.assertTrue(torch.allclose(product, I2, atol=1e-12))

    def test_signal_operator_zero_is_identity(self):
        Wt = signal_operator(torch.tensor([0.0], dtype=torch.float64))[0]
        I2 = torch.eye(2, dtype=torch.complex128)
        self.assertTrue(torch.allclose(Wt, I2, atol=1e-14))

    def test_signal_operator_pi_is_neg_i_sigma_x(self):
        """signal_operator(pi) = [[0, -i], [-i, 0]] = -i sigma_x."""
        Wt = signal_operator(torch.tensor([math.pi], dtype=torch.float64))[0]
        expected = torch.tensor([[0, -1j], [-1j, 0]], dtype=torch.complex128)
        self.assertTrue(torch.allclose(Wt, expected, atol=1e-12))


class TestWaitTimeSignalOperator(unittest.TestCase):
    """
    Waiting t = pi/(2*Delta_0) with H = 1/2*(Delta_0+delta)*sigma_z
    should produce a Z-rotation of angle theta(delta) = pi/2*(1+delta/Delta_0).

    In QSP basis (swap x<->z), this is R_x(theta) = W(theta).
    """

    def test_wait_time_produces_correct_theta(self):
        t_wait = math.pi / (2 * DELTA_0)
        for delta_mhz in [-150.0, -50.0, 0.0, 50.0, 150.0]:
            delta = 2 * math.pi * delta_mhz
            phys_angle = (DELTA_0 + delta) * t_wait
            expected_theta = delta_to_theta(
                torch.tensor(delta, dtype=torch.float64), DELTA_0
            ).item()
            self.assertAlmostEqual(
                phys_angle, expected_theta, places=10,
                msg=f"Wait time rotation angle wrong for delta={delta_mhz} MHz",
            )


class TestControlOperator(unittest.TestCase):
    """Verify build_qsp_unitary control operator behavior."""

    def test_ideal_control_is_rz_at_zero_delta(self):
        """When delta=0, the noisy control Rz(phi; delta=0) should equal ideal Rz(phi)."""
        for phi_val in [0.5, 1.0, -0.7, math.pi]:
            phi = torch.tensor([phi_val], dtype=torch.float64)
            delta = torch.tensor([0.0], dtype=torch.float64)

            # build_qsp_unitary with K=0 (single phase) produces just
            # the control operator applied once (no signal in between)
            U_noisy = build_qsp_unitary(
                phi, delta, DELTA_0, OMEGA_MAX,
            )
            U_ideal = Rz(phi)

            self.assertTrue(
                torch.allclose(U_noisy[0], U_ideal[0], atol=1e-10),
                f"Control at delta=0 should be ideal Rz for phi={phi_val}",
            )

    def test_control_is_unitary(self):
        """The noisy control operator should always produce a unitary."""
        phi = torch.randn(15, dtype=torch.float64)
        deltas = torch.linspace(-DELTA_0 * 0.9, DELTA_0 * 0.9, 20, dtype=torch.float64)
        U = build_qsp_unitary(phi, deltas, DELTA_0, OMEGA_MAX)
        I2 = torch.eye(2, dtype=torch.complex128).unsqueeze(0).expand(20, -1, -1)
        product = torch.bmm(U, U.conj().transpose(-1, -2))
        self.assertTrue(
            torch.allclose(product, I2, atol=1e-10),
            "build_qsp_unitary produced non-unitary result",
        )


class TestRuntimeComputation(unittest.TestCase):
    """Verify get_control_runtime matches manual calculation."""

    def test_runtime_formula(self):
        """T = K * tau + sum(|phi|) / Omega, tau = pi/(2*Delta_0)."""
        K = 70
        phi = torch.randn(K + 1, dtype=torch.float64)
        cfg = TrainConfig(
            Omega_max=OMEGA_MAX, Delta_0=DELTA_0,
            robustness_window=SIGNAL_WINDOW, K=K,
        )
        runtime = get_control_runtime(phi, cfg)

        tau = math.pi / (2 * DELTA_0)
        expected = K * tau + phi.abs().sum().item() / OMEGA_MAX
        self.assertAlmostEqual(runtime, expected, places=12)

    def test_runtime_matches_pulse_schedule(self):
        """Runtime from get_control_runtime should match sum of pulse schedule durations."""
        K = 10
        phi = torch.randn(K + 1, dtype=torch.float64)
        cfg = TrainConfig(
            Omega_max=OMEGA_MAX, Delta_0=DELTA_0,
            robustness_window=SIGNAL_WINDOW, K=K,
        )

        # Build pulse schedule the same way streamlit_app.py does
        tau_us = math.pi / (2 * cfg.Delta_0)
        t_rows = []
        for i, pv in enumerate(phi.tolist()):
            t_rows.append(abs(pv) / cfg.Omega_max)
            if i != len(phi) - 1:
                t_rows.append(tau_us)

        schedule_runtime = sum(t_rows)
        computed_runtime = get_control_runtime(phi, cfg)
        self.assertAlmostEqual(schedule_runtime, computed_runtime, places=12)


class TestHadamardBasisChange(unittest.TestCase):
    """
    The Hadamard H swaps QSP basis (z-tilde, x-tilde) <-> physical (x, z).
    Target in QSP basis: Rz(alpha) = diag(e^{-i alpha/2}, e^{+i alpha/2})
    Target in physical basis: Rx(alpha) = H @ Rz(alpha) @ H
    """

    def test_hadamard_converts_rz_to_rx(self):
        H_mat = torch.tensor(
            [[1, 1], [1, -1]], dtype=torch.complex128
        ) / math.sqrt(2)

        for alpha in [0.0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3]:
            Rz_alpha = torch.tensor(
                [[np.exp(-1j * alpha / 2), 0],
                 [0, np.exp(1j * alpha / 2)]],
                dtype=torch.complex128,
            )
            Rx_alpha = torch.tensor(
                [[math.cos(alpha / 2), -1j * math.sin(alpha / 2)],
                 [-1j * math.sin(alpha / 2), math.cos(alpha / 2)]],
                dtype=torch.complex128,
            )
            converted = H_mat @ Rz_alpha @ H_mat
            self.assertTrue(
                torch.allclose(converted, Rx_alpha, atol=1e-12),
                f"H @ Rz({alpha}) @ H != Rx({alpha})",
            )


class TestIdealQSPLimit(unittest.TestCase):
    """
    When Omega >> Delta_0 (fast control limit), the noisy control operator
    approaches the ideal Rz (detuning contribution becomes negligible).
    We verify by comparing build_qsp_unitary at very large Omega against
    a manually computed ideal QSP product.
    """

    def test_large_omega_approaches_ideal(self):
        torch.manual_seed(42)
        K = 5
        phi = torch.randn(K + 1, dtype=torch.float64) * 0.3
        delta = torch.tensor([2 * math.pi * 30.0], dtype=torch.float64)
        theta = delta_to_theta(delta, DELTA_0)

        # Very large Omega => detuning effect in control is negligible
        Omega_huge = 2 * math.pi * 1e6
        U_noisy = build_qsp_unitary(phi, delta, DELTA_0, Omega_huge)

        # Manually build ideal QSP matching build_qsp_unitary ordering:
        # U = Rz(phi_K) @ W @ Rz(phi_{K-1}) @ ... @ W @ Rz(phi_0)
        W_mat = signal_operator(theta)  # (1, 2, 2)
        U_ideal = Rz(phi[K:K+1])  # (1, 2, 2)
        for j in range(K - 1, -1, -1):
            U_ideal = bmm(U_ideal, W_mat)
            U_ideal = bmm(U_ideal, Rz(phi[j:j + 1]))

        self.assertTrue(
            torch.allclose(U_noisy[0], U_ideal[0], atol=1e-4),
            "build_qsp_unitary at huge Omega should match ideal QSP product",
        )


class TestFidelityComputation(unittest.TestCase):
    """Test the fidelity function from qsp_fit_relaxed."""

    def test_identity_has_fidelity_one(self):
        """A phi that produces identity at all deltas should have F=1 for alpha=0."""
        # Zero-length sequence: K=0, phi=[0] => U = Rz(0) = I
        phi = torch.tensor([0.0], dtype=torch.float64)
        delta_vals = torch.tensor([0.0], dtype=torch.float64)
        alpha_vals = torch.tensor([0.0], dtype=torch.float64)
        cfg = TrainConfig(
            Omega_max=OMEGA_MAX, Delta_0=DELTA_0,
            robustness_window=SIGNAL_WINDOW, K=0,
        )
        F = qsp_fidelity(phi, delta_vals, alpha_vals, cfg)
        self.assertGreater(F, 0.999, f"Identity QSP should give F~1, got {F}")

    def test_fidelity_bounded_zero_one(self):
        """Fidelity should always be in [0, 1]."""
        torch.manual_seed(123)
        phi = torch.randn(11, dtype=torch.float64)
        delta_vals = torch.tensor([-100.0, 100.0], dtype=torch.float64) * 2 * math.pi
        alpha_vals = torch.tensor([math.pi / 2, math.pi], dtype=torch.float64)
        cfg = TrainConfig(
            Omega_max=OMEGA_MAX, Delta_0=DELTA_0,
            robustness_window=SIGNAL_WINDOW, K=10,
        )
        F = qsp_fidelity(phi, delta_vals, alpha_vals, cfg)
        self.assertGreaterEqual(F, 0.0)
        self.assertLessEqual(F, 1.0 + 1e-6)


class TestTrainingConvergence(unittest.TestCase):
    """
    Smoke test: a short training run on a simple target should reduce loss.
    Not meant to reach high fidelity, just to verify the training loop works.
    """

    def test_training_reduces_loss(self):
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)
        cfg = TrainConfig(
            Omega_max=OMEGA_MAX,
            Delta_0=DELTA_0,
            robustness_window=SIGNAL_WINDOW,
            K=4,
            steps=500,
            lr=0.05,
            device="cpu",
            out_dir="test_plots_tmp",
        )
        os.makedirs(cfg.out_dir, exist_ok=True)

        # Single peak, identity gate (alpha=0) — easiest target
        delta_vals = torch.tensor([0.0], dtype=torch.float64)
        alpha_vals = torch.tensor([0.0], dtype=torch.float64)

        phi, loss, fid = train(
            cfg, delta_vals, alpha_vals,
            sample_size=256, verbose=False,
        )

        self.assertLess(loss, 0.5, f"Loss after 500 steps should be < 0.5, got {loss}")
        self.assertEqual(phi.shape[0], cfg.K + 1)

        # Cleanup
        import shutil
        if os.path.exists(cfg.out_dir):
            shutil.rmtree(cfg.out_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  SCALING LAW VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

sigma_x_np = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z_np = np.array([[1, 0], [0, -1]], dtype=complex)


def runtime_independent(pulse_df: pd.DataFrame) -> float:
    """Return total runtime in microseconds (sum of t column)."""
    return float(pulse_df["t (us)"].sum())


def fidelity_independent(
    pulse_df: pd.DataFrame,
    delta: list,
    alpha: list,
    robustness_window: float,
    sample_size: int = 2000,
) -> float:
    """
    Compute average gate fidelity over sampled detunings.

    For each (delta_i, alpha_i) pair, sample_size detuning values are drawn
    uniformly from [delta_i - robustness_window, delta_i + robustness_window].

    Uses the analytic 2×2 matrix exponential and batches all delta_samples
    simultaneously, removing the per-sample Python loop.

    For each pulse step the propagator is:
        V_j = exp[-i*(Omega_x_j*sigma_x + (Omega_z_j+delta_s)*sigma_z)*t_j/2]
            = cos(r)*I - i*sin(r)/r * (a*sigma_x + b*sigma_z)
    where a = Omega_x_j*t_j/2, b = (Omega_z_j+delta_s)*t_j/2, r = sqrt(a^2+b^2).
    Column names "Omega_x/z (2pi MHz)" are tried first, falling back to "H_x/z (2pi MHz)".

    Target: R_x(alpha_i) = exp(-i * sigma_x * alpha_i / 2)
    Fidelity: (|Tr(U^dag @ R_x)|^2 + 2) / 6
    """
    assert len(delta) == len(alpha)

    ts = pulse_df["t (us)"].to_numpy()
    Omega_xs = (pulse_df["Omega_x (2pi MHz)"] if "Omega_x (2pi MHz)" in pulse_df.columns
                else pulse_df["H_x (2pi MHz)"]).to_numpy()
    Omega_zs = (pulse_df["Omega_z (2pi MHz)"] if "Omega_z (2pi MHz)" in pulse_df.columns
                else pulse_df["H_z (2pi MHz)"]).to_numpy()

    all_fidelities = []
    rng = np.random.default_rng(seed=42)

    for delta_i, alpha_i in zip(delta, alpha):
        delta_s = rng.uniform(
            delta_i - robustness_window,
            delta_i + robustness_window,
            size=sample_size,
        )  # [S]

        R_x = (
            np.cos(alpha_i / 2) * np.eye(2, dtype=complex)
            - 1j * np.sin(alpha_i / 2) * sigma_x_np
        )

        # U[s] = V_L @ ... @ V_1 for all S samples simultaneously
        U = np.broadcast_to(np.eye(2, dtype=complex), (sample_size, 2, 2)).copy()

        for t_j, omegax_j, omegaz_j in zip(ts, Omega_xs, Omega_zs):
            # exp[-1j * (omegax_j*sigma_x + (omegaz_j+delta_s)*sigma_z) * t_j/2]
            # = cos(r)*I - i*sin(r)/r * (a*sigma_x + b*sigma_z)
            # where a = omegax_j*t_j/2, b = (omegaz_j+delta_s)*t_j/2, r = sqrt(a^2+b^2)
            a = 2 * math.pi * omegax_j * t_j / 2                    # scalar
            b = 2 * math.pi * (omegaz_j + delta_s) * t_j / 2        # [S]
            r = np.sqrt(a**2 + b**2)                   # [S]
            sinc_r = np.where(r > 1e-15, np.sin(r) / r, 1.0)  # [S]
            c = np.cos(r)                               # [S]

            V = np.empty((sample_size, 2, 2), dtype=complex)
            V[:, 0, 0] = c - 1j * b * sinc_r
            V[:, 1, 1] = c + 1j * b * sinc_r
            V[:, 0, 1] = -1j * a * sinc_r              # a is scalar, broadcasts
            V[:, 1, 0] = -1j * a * sinc_r
            U = np.matmul(V, U)                         # [S, 2, 2]

        # Tr(U^dag @ R_x) vectorized: einsum('sji,ji->s', conj(U), R_x)
        traces = np.einsum('sji,ji->s', np.conj(U), R_x)
        all_fidelities.append((np.abs(traces) ** 2 + 2) / 6)

    return float(np.concatenate(all_fidelities).mean())



def _load_input(data_dir: Path, omega_max: int, K: int, trial: int):
    path = data_dir / f"Omega_max{omega_max}_K{K}_trial{trial}_input.csv"
    df = pd.read_csv(path)
    return df["delta (MHz)"].tolist(), df["alpha (pi rad)"].tolist()


class TestScalingLawResultsBase(unittest.TestCase):
    """Base class for scaling law verification tests. Subclasses set FILE_DIR."""

    FILE_DIR: str  # must be set by subclass
    TOLERANCE_FIDELITY = 1e-2 # up to 1% fidelity error
    TOLERANCE_RUNTIME = 1e-6

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(os.path.join(cls.FILE_DIR, "data"))
        cls.results_csv = Path(os.path.join(cls.FILE_DIR, "scaling_law_fidelity.csv"))
        if not cls.results_csv.exists():
            raise unittest.SkipTest(
                f"Scaling law results not found at {cls.results_csv}. "
                f"Run scaling_law.py first or set FILE_DIR."
            )
        cls.results = pd.read_csv(cls.results_csv)

    def test_runtime_matches(self):
        missing, failures = [], []
        for _, row in tqdm(list(self.results.iterrows()), desc=f"Runtime check [{self.FILE_DIR}]"):
            omega_max = int(row["Omega_max (MHz)"])
            K = int(row["K"])
            trial = int(row["trial"])
            exp_rt = float(row["Runtime (us)"])

            pulse_path = self.data_dir / f"Omega_max{omega_max}_K{K}_trial{trial}.csv"
            if not pulse_path.exists():
                missing.append(f"Omega_max={omega_max}, K={K}, trial={trial}")
                continue

            pulse_df = pd.read_csv(pulse_path)
            computed_rt = runtime_independent(pulse_df)
            label = f"Omega_max={omega_max}, K={K}, trial={trial}"

            if abs(computed_rt - exp_rt) > self.TOLERANCE_RUNTIME:
                failures.append(
                    f"  {label}: expected={exp_rt:.9f} got={computed_rt:.9f} "
                    f"diff={computed_rt - exp_rt:+.9f}"
                )

        if failures:
            msg = f"RUNTIME MISMATCHES ({len(failures)}/{len(self.results)}):\n"
            msg += "\n".join(failures[:10])
            if len(failures) > 10:
                msg += f"\n  ... and {len(failures) - 10} more"
            self.fail(msg)

    def test_fidelity_matches(self):
        missing, failures = [], []
        for _, row in tqdm(list(self.results.iterrows()), desc=f"Fidelity check [{self.FILE_DIR}]"):
            omega_max = int(row["Omega_max (MHz)"])
            K = int(row["K"])
            trial = int(row["trial"])
            exp_fid = float(row["Gate Fidelity"])

            pulse_path = self.data_dir / f"Omega_max{omega_max}_K{K}_trial{trial}.csv"
            input_path = self.data_dir / f"Omega_max{omega_max}_K{K}_trial{trial}_input.csv"
            if not pulse_path.exists() or not input_path.exists():
                missing.append(f"Omega_max={omega_max}, K={K}, trial={trial}")
                continue

            pulse_df = pd.read_csv(pulse_path)
            delta, alpha_pirad = _load_input(self.data_dir, omega_max, K, trial)
            alpha_rad = [a * np.pi for a in alpha_pirad]

            computed_fid = fidelity_independent(
                pulse_df, delta, alpha_rad, robustness_window=10.0
            )
            label = f"Omega_max={omega_max}, K={K}, trial={trial}"

            if abs(computed_fid - exp_fid) > self.TOLERANCE_FIDELITY:
                failures.append(
                    f"  {label}: expected={exp_fid:.9f} got={computed_fid:.9f} "
                    f"diff={computed_fid - exp_fid:+.9f}"
                )

        if failures:
            msg = f"FIDELITY MISMATCHES ({len(failures)}/{len(self.results)}):\n"
            msg += "\n".join(failures[:10])
            if len(failures) > 10:
                msg += f"\n  ... and {len(failures) - 10} more"
            self.fail(msg)


class TestScalingLawSmall(TestScalingLawResultsBase):
    """Scaling law verification on the small dataset (scaling_law_small)."""
    FILE_DIR = "scaling_law_small"


class TestScalingLawLarge(TestScalingLawResultsBase):
    """Scaling law verification on the large dataset (scaling_law_results)."""
    FILE_DIR = "scaling_law_results"


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run QSP test suite.")
    parser.add_argument(
        "--f",
        type=str,
        default="all",
        choices=["implementation", "scaling_small", "scaling_large", "scaling", "all"],
        help=(
            "Which test suite to run: "
            "'implementation' (unit tests), "
            "'scaling_small' (scaling_law_small dataset), "
            "'scaling_large' (scaling_law_results dataset), "
            "'scaling' (both scaling datasets), "
            "'all' (everything)."
        ),
    )
    args = parser.parse_args()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    implementation_classes = [
        TestDeltaThetaMapping,
        TestSignalOperator,
        TestWaitTimeSignalOperator,
        TestControlOperator,
        TestRuntimeComputation,
        TestHadamardBasisChange,
        TestIdealQSPLimit,
        TestFidelityComputation,
        TestTrainingConvergence,
    ]

    if args.f in ("implementation", "all"):
        for cls in implementation_classes:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    if args.f in ("scaling_small", "scaling", "all"):
        suite.addTests(loader.loadTestsFromTestCase(TestScalingLawSmall))

    if args.f in ("scaling_large", "scaling", "all"):
        suite.addTests(loader.loadTestsFromTestCase(TestScalingLawLarge))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
