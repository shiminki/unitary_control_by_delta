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


def _evaluate_fit(fit, fit_type, alpha_grid):
    """Evaluate fitted amplitude at the given alpha values.

    Parameters
    ----------
    fit : dict (fourier) or tuple (polyfit)
    fit_type : 'fourier_fit' or 'polyfit'
    alpha_grid : 1-D array of alpha values
    """
    if fit_type == "fourier_fit":
        coeffs = fit["coeffs"]
        n_fourier = fit["n_fourier"]
        design = [np.ones(len(alpha_grid))]
        for k in range(1, n_fourier + 1):
            design.append(np.cos(k * alpha_grid / 2))
            design.append(np.sin(k * alpha_grid / 2))
        return np.column_stack(design) @ coeffs
    else:
        _, coeffs, _ = fit
        return np.polyval(coeffs, alpha_grid)


# ─────────────────────────────────────────────────────────────────────────────
#  PCA plots (unified for both fit types)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_overview(
    alpha_grid, phi_matrix, S, explained_ratio, cumulative,
    n_dof, amplitudes, fits, fit_type, peak_index, out_dir,
    explained_var_threshold=0.999,
):
    """Generate 4-panel PCA overview plot for either fit type.

    Panels: singular values, residual variance, amplitude functions with
    fit overlay, sample phi trajectories.

    Saves to out_dir/pca_overview/pca_peak{N}.png
    """
    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polynomial"

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
    ax.set_title(f"Residual Variance (Linear Components = {n_dof})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Amplitude functions A_j(alpha) with fit overlay
    ax = axes[1, 0]
    for j in range(min(n_dof, 5)):
        ax.plot(alpha_grid / math.pi, amplitudes[:, j], label=f"A_{j}")
        fitted_vals = _evaluate_fit(fits[j], fit_type, alpha_grid)
        ax.plot(alpha_grid / math.pi, fitted_vals, "--", alpha=0.7)
    ax.set_xlabel(r"$\alpha / \pi$")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Amplitude functions (solid=data, dashed={fit_label} fit)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Sample phi trajectories
    ax = axes[1, 1]
    n_show = min(5, phi_matrix.shape[0])
    indices = np.linspace(0, phi_matrix.shape[0] - 1, n_show, dtype=int)
    for idx in indices:
        alpha_val = alpha_grid[idx]
        ax.plot(phi_matrix[idx], label=f"{alpha_val/math.pi:.2f}pi", alpha=0.7)
    ax.set_xlabel("Phase index j")
    ax.set_ylabel("phi_j")
    ax.set_title("Sample phi vectors")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"PCA Analysis for Peak {peak_index} ({fit_label} Fit)", fontsize=14)
    plt.tight_layout()
    overview_dir = os.path.join(out_dir, "pca_overview")
    os.makedirs(overview_dir, exist_ok=True)
    plt.savefig(os.path.join(overview_dir, f"pca_peak{peak_index}.png"), dpi=150)
    plt.close()


def plot_pca_components(amplitudes, alpha_grid, peak_index, out_dir):
    """Plot phi(alpha) projected onto the first two principal components, colored by alpha."""
    if amplitudes.shape[1] < 2:
        print("  Fewer than 2 PCA components, skipping PCA component plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    pc1, pc2 = amplitudes[:, 0], amplitudes[:, 1]

    ax.plot(pc1, pc2, "-", color="gray", alpha=0.3, linewidth=0.5)

    alpha_max = alpha_grid.max()
    sc = ax.scatter(
        pc1, pc2,
        c=alpha_grid, cmap="hsv", s=20, alpha=0.8,
        vmin=0, vmax=alpha_max,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\alpha$ (rad)")
    period_in_pi = round(alpha_max / math.pi)
    if period_in_pi <= 2:
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    else:
        cbar.set_ticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA Components of $\\phi(\\alpha)$ trajectory (Peak {peak_index})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pca_comp_dir = os.path.join(out_dir, "pca_components")
    os.makedirs(pca_comp_dir, exist_ok=True)
    plt.savefig(os.path.join(pca_comp_dir, f"pca_components_peak{peak_index}.png"), dpi=150)
    plt.close()


def plot_amplitude_components(
    alpha_grid, amplitudes, fits, fit_type, n_dof, peak_index, out_dir,
):
    """Per-component amplitude fit subplots for either fit type.

    Saves to out_dir/amplitudes/amplitude_fits_peak{N}.png
    """
    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polynomial"

    n_cols = min(3, n_dof)
    n_rows = math.ceil(n_dof / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    alpha_fine = np.linspace(alpha_grid.min(), alpha_grid.max(), 500)

    for k in range(n_dof):
        row_idx, col_idx = divmod(k, n_cols)
        ax = axes[row_idx, col_idx]

        fitted_fine = _evaluate_fit(fits[k], fit_type, alpha_fine)

        ax.plot(alpha_grid / math.pi, amplitudes[:, k],
                ".", color="C0", markersize=3, alpha=0.5, label="data")
        ax.plot(alpha_fine / math.pi, fitted_fine,
                "-", color="C1", linewidth=1.8, label=f"{fit_label} fit")

        ax.set_xlabel(r"$\alpha\,/\,\pi$")
        ax.set_ylabel(f"$A_{{{k}}}(\\alpha)$")

        if fit_type == "fourier_fit":
            residual = fits[k]["residual"]
            n_fourier = fits[k]["n_fourier"]
            ax.set_title(f"$k={k}$,  n_fourier={n_fourier},  res={residual:.2e}")
        else:
            deg, _, r2 = fits[k]
            ax.set_title(f"$k={k}$,  deg={deg},  $R^2={r2:.4f}$")

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for k in range(n_dof, n_rows * n_cols):
        row_idx, col_idx = divmod(k, n_cols)
        axes[row_idx, col_idx].set_visible(False)

    plt.suptitle(
        f"{fit_label} Fit of $A_k(\\alpha)$ — Peak {peak_index}",
        fontsize=13,
    )
    plt.tight_layout()
    amplitudes_dir = os.path.join(out_dir, "amplitudes")
    os.makedirs(amplitudes_dir, exist_ok=True)
    plt.savefig(os.path.join(amplitudes_dir, f"amplitude_fits_peak{peak_index}.png"), dpi=150)
    plt.close()


def plot_tsne(phi_matrix, alpha_grid, peak_index, out_dir):
    """Generate t-SNE visualization of phi(alpha) trajectory colored by alpha."""
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

    # Draw trajectory line connecting consecutive alpha samples
    ax.plot(phi_2d[:, 0], phi_2d[:, 1], "-", color="gray", alpha=0.3, linewidth=0.5)

    # Scatter colored by alpha (cyclic colormap)
    alpha_max = alpha_grid.max()
    sc = ax.scatter(
        phi_2d[:, 0], phi_2d[:, 1],
        c=alpha_grid, cmap="hsv", s=20, alpha=0.8,
        vmin=0, vmax=alpha_max,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\alpha$ (rad)")
    period_in_pi = round(alpha_max / math.pi)
    if period_in_pi <= 2:
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    else:
        cbar.set_ticks([0, np.pi, 2 * np.pi, 3 * np.pi, 4 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$", r"$3\pi$", r"$4\pi$"])

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of $\\phi(\\alpha)$ trajectory (Peak {peak_index})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_dir = os.path.join(out_dir, "tsne")
    os.makedirs(tsne_dir, exist_ok=True)
    plt.savefig(os.path.join(tsne_dir, f"tsne_peak{peak_index}.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Comparative plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparative_pulses(
    net, pca_result, alpha_vals=None, out_dir="nn_pulse_output",
    fit_type="fourier_fit", n_basis=2,
):
    """Plot H_x(t) and H_z(t) for NN output vs fit reconstruction.

    Parameters
    ----------
    fit_type : 'fourier_fit' or 'polyfit'
        Which reconstruction to compare against.
    n_basis : int
        Number of PCA basis vectors to use in reconstruction.

    Saves to out_dir/comparative_pulses/.
    """
    from .pca import reconstruct_phi_analytical, reconstruct_phi_polyfit

    if alpha_vals is None:
        alpha_vals = [math.pi / 4, math.pi / 2, math.pi]

    net.eval()

    if fit_type == "fourier_fit":
        reconstruct_fn = lambda alpha: reconstruct_phi_analytical(pca_result, alpha, n_basis=n_basis)
        fit_label = f"Fourier (d={n_basis})"
    else:
        reconstruct_fn = lambda alpha: reconstruct_phi_polyfit(pca_result, alpha, n_basis=n_basis)
        fit_label = f"Polyfit (d={n_basis})"

    n_alpha = len(alpha_vals)
    fig, axes = plt.subplots(n_alpha, 2, figsize=(16, 4 * n_alpha), squeeze=False)

    for row, alpha in enumerate(alpha_vals):
        with torch.no_grad():
            phi_nn = net(torch.tensor([alpha], dtype=torch.float64)).squeeze(0)
        pdf_nn = phi_to_pulse_df(phi_nn, net.Omega_mhz, net.Delta_0_mhz)
        t_nn, hx_nn, hz_nn = _pulse_df_to_time_series(pdf_nn)

        phi_fit = reconstruct_fn(alpha)
        pdf_fit = phi_to_pulse_df(phi_fit, net.Omega_mhz, net.Delta_0_mhz)
        t_fit, hx_fit, hz_fit = _pulse_df_to_time_series(pdf_fit)

        alpha_label = f"{alpha/math.pi:.2f}"

        # H_x
        ax = axes[row, 0]
        ax.step(t_nn, np.r_[hx_nn, hx_nn[-1]], where="post",
                label="NN (actual)", linewidth=1.5)
        ax.step(t_fit, np.r_[hx_fit, hx_fit[-1]], where="post",
                label=fit_label, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(r"$\Omega_x(t)$ (2$\pi$ MHz)")
        ax.set_ylim(-1.1 * net.Omega_mhz, 1.1 * net.Omega_mhz)
        ax.set_title(rf"$\alpha = {alpha_label}\pi$  —  $\Omega_x(t)$")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == n_alpha - 1:
            ax.set_xlabel("Time (ns)")

        # H_z
        ax = axes[row, 1]
        ax.step(t_nn, np.r_[hz_nn, hz_nn[-1]], where="post",
                label="NN (actual)", linewidth=1.5)
        ax.step(t_fit, np.r_[hz_fit, hz_fit[-1]], where="post",
                label=fit_label, linestyle="--", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(r"$\Omega_z(t)$ (2$\pi$ MHz)")
        ax.set_ylim(-10, net.Delta_0_mhz + 20)
        ax.set_title(rf"$\alpha = {alpha_label}\pi$  —  $\Omega_z(t)$")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == n_alpha - 1:
            ax.set_xlabel("Time (ns)")

    plt.tight_layout()
    comp_pulses_dir = os.path.join(out_dir, "comparative_pulses")
    os.makedirs(comp_pulses_dir, exist_ok=True)
    plt.savefig(os.path.join(comp_pulses_dir,
                f"comparative_pulses_peak{net.peak_index}.png"), dpi=150)
    plt.close()


def plot_comparative_matrix_elements(
    net, pca_result, alpha_vals=None, out_dir="nn_pulse_output",
    fit_type="fourier_fit", n_basis=2,
):
    """Plot Re(u_00) and Im(u_01) vs detuning for NN vs fit reconstruction.

    Parameters
    ----------
    fit_type : 'fourier_fit' or 'polyfit'
        Which reconstruction to compare against.
    n_basis : int
        Number of PCA basis vectors to use in reconstruction.

    Saves to out_dir/comparative_matrix_element/.
    Returns a dict mapping alpha -> {"fidelity_nn": float, "fidelity_fit": float}.
    """
    from .pca import reconstruct_phi_analytical, reconstruct_phi_polyfit

    if alpha_vals is None:
        alpha_vals = [math.pi / 4, math.pi / 2, math.pi]

    net.eval()

    if fit_type == "fourier_fit":
        reconstruct_fn = lambda alpha: reconstruct_phi_analytical(pca_result, alpha, n_basis=n_basis)
        fit_label = f"Fourier (d={n_basis})"
    else:
        reconstruct_fn = lambda alpha: reconstruct_phi_polyfit(pca_result, alpha, n_basis=n_basis)
        fit_label = f"Polyfit (d={n_basis})"

    n_alpha = len(alpha_vals)
    fig, axes = plt.subplots(n_alpha, 1, figsize=(10, 5 * n_alpha), squeeze=False)

    sigma_mhz = net.robustness_window_mhz
    delta_centers = net.delta_centers_mhz
    peak_idx = net.peak_index

    fidelity_results = {}

    for row, alpha in enumerate(alpha_vals):
        ax = axes[row, 0]

        with torch.no_grad():
            phi_nn = net(torch.tensor([alpha], dtype=torch.float64)).squeeze(0)
        dm_nn, u00_nn, u01_nn = _compute_matrix_elements(
            phi_nn, net.Delta_0_ang, net.Omega_ang)

        phi_fit = reconstruct_fn(alpha)
        dm_fit, u00_fit, u01_fit = _compute_matrix_elements(
            phi_fit, net.Delta_0_ang, net.Omega_ang)

        # Compute fidelities
        fid_nn = compute_fidelity_for_phi(
            phi_nn, peak_idx, alpha, delta_centers,
            net.Omega_mhz, net.Delta_0_mhz, sigma_mhz)
        fid_fit = compute_fidelity_for_phi(
            phi_fit, peak_idx, alpha, delta_centers,
            net.Omega_mhz, net.Delta_0_mhz, sigma_mhz)
        fidelity_results[alpha] = {"fidelity_nn": fid_nn, "fidelity_fit": fid_fit}

        ax.plot(dm_nn, u00_nn.real, color="C0", linewidth=1.2,
                linestyle="--", label="Re(u_00) NN")
        ax.plot(dm_nn, u01_nn.imag, color="C1", linewidth=1.2,
                linestyle="--", label="Im(u_01) NN")
        ax.plot(dm_fit, u00_fit.real, color="C2", linewidth=1.2,
                alpha=0.7, label=f"Re(u_00) {fit_label}")
        ax.plot(dm_fit, u01_fit.imag, color="C3", linewidth=1.2,
                alpha=0.7, label=f"Im(u_01) {fit_label}")

        for j, dc in enumerate(delta_centers):
            alpha_j = alpha if j == peak_idx else 0.0
            ax.hlines(y=np.cos(alpha_j / 2),
                      xmin=dc - sigma_mhz, xmax=dc + sigma_mhz,
                      colors="red", linestyles="dotted", linewidth=1.5,
                      label="target Re(u_00)" if (row == 0 and j == 0) else None)
            ax.hlines(y=-np.sin(alpha_j / 2),
                      xmin=dc - sigma_mhz, xmax=dc + sigma_mhz,
                      colors="green", linestyles="dotted", linewidth=1.5,
                      label="target Im(u_01)" if (row == 0 and j == 0) else None)
            ax.axvspan(dc - sigma_mhz, dc + sigma_mhz, color="gray", alpha=0.1)

        alpha_label = f"{alpha/math.pi:.2f}"
        ax.set_title(
            rf"$\alpha = {alpha_label}\pi$  —  "
            rf"Fidelity: NN = {fid_nn:.4f}, {fit_label} = {fid_fit:.4f}")
        ax.set_ylabel("Matrix Element")
        ax.set_ylim(-1.3, 1.3)
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        if row == n_alpha - 1:
            ax.set_xlabel(r"Detuning $\delta$ (MHz)")

    plt.tight_layout()
    comp_me_dir = os.path.join(out_dir, "comparative_matrix_element")
    os.makedirs(comp_me_dir, exist_ok=True)
    plt.savefig(os.path.join(
        comp_me_dir, f"comparative_matrix_elements_peak{peak_idx}.png"), dpi=150)
    plt.close()

    return fidelity_results
