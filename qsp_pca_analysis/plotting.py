"""All visualization: PCA plots, t-SNE, comparative pulse/matrix element plots."""

import math
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from single_pulse_optimization_QSP.qsp_fit_x_rotation import build_qsp_unitary

from .pulse import phi_to_pulse_df
from .evaluation import compute_fidelity_for_phi


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  PCA plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_results(
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


def plot_tsne(phi_matrix, theta_grid, peak_index, out_dir):
    """Generate t-SNE visualization of phi(theta) trajectory colored by theta."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  sklearn not available, skipping t-SNE plot")
        return

    try:
        perplexity = min(30, len(phi_matrix) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, method='exact')
        phi_2d = tsne.fit_transform(phi_matrix)
    except Exception as e:
        print(f"  t-SNE failed ({e}), skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw trajectory line connecting consecutive theta samples
    ax.plot(phi_2d[:, 0], phi_2d[:, 1], "-", color="gray", alpha=0.3, linewidth=0.5)

    # Scatter colored by theta (cyclic colormap)
    sc = ax.scatter(
        phi_2d[:, 0], phi_2d[:, 1],
        c=theta_grid, cmap="hsv", s=20, alpha=0.8,
        vmin=0, vmax=2 * math.pi,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\theta$ (rad)")
    cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of $\\phi(\\theta)$ trajectory (Peak {peak_index})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"tsne_peak{peak_index}.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Comparative plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparative_pulses(net, pca_result, theta_vals=None, out_dir="nn_pulse_output"):
    """Plot H_x(t) and H_z(t) for NN output vs PCA-analytical reconstruction.

    For each theta in theta_vals, produces a row with two subplots
    (H_x on left, H_z on right) comparing the actual NN pulse with
    the analytical PCA-reconstructed pulse.
    """
    # Deferred import to break pca <-> plotting circular dependency
    from .pca import reconstruct_phi_analytical

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


def plot_comparative_matrix_elements(net, pca_result, theta_vals=None, out_dir="nn_pulse_output"):
    """Plot Re(u_00) and Im(u_01) vs detuning for NN vs PCA-analytical.

    For each theta, shows the matrix elements of the physical-basis unitary
    across all detunings, comparing the actual NN pulse with the analytical
    PCA-reconstructed pulse.  Target windows are overlaid.

    Returns a dict mapping theta -> {"fidelity_nn": float, "fidelity_pca": float}.
    """
    # Deferred import to break pca <-> plotting circular dependency
    from .pca import reconstruct_phi_analytical

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
