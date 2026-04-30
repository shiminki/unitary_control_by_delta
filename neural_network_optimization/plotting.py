"""Figures for the refactored pipeline.

Three scaling-law / per-model figures plus the four PCA figures.  All functions
save to the supplied ``out_path`` (or directory for PCA) and return nothing
useful besides that.
"""

import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from .inference import compute_fidelity, compute_runtime, generate_phi
from .model import PulseNet, phi_to_pulse_df
from .physics import build_qsp_unitary


# ─────────────────────────────────────────────────────────────────────────────
#  Matrix element vs detuning
# ─────────────────────────────────────────────────────────────────────────────

def plot_matrix_element(
    model: PulseNet,
    alpha,
    out_path: str,
    n_delta: int = 1024,
    sample_size_for_fidelity: int = 5000,
):
    """Plot Re(u_00), Im(u_01) vs detuning with target windows at each peak."""
    model.eval()
    dev = next(model.parameters()).device

    if not isinstance(alpha, torch.Tensor):
        alpha_t = torch.tensor(alpha, dtype=torch.float64, device=dev)
    else:
        alpha_t = alpha.to(dtype=torch.float64, device=dev)
    assert alpha_t.shape[-1] == model.N_peaks

    with torch.no_grad():
        phi = model(alpha_t.unsqueeze(0)).squeeze(0).detach().cpu()

    Delta_0_ang = model.Delta_0_ang
    Omega_ang = model.Omega_ang
    delta_range = torch.linspace(-Delta_0_ang, Delta_0_ang, n_delta, dtype=torch.float64)
    U = build_qsp_unitary(phi, delta_range, Delta_0_ang, Omega_ang)

    # QSP -> physical basis via Hadamard sandwich
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    U = H @ U @ H
    u00 = U[:, 0, 0].detach().cpu()
    u01 = U[:, 0, 1].detach().cpu()

    delta_mhz = (delta_range / (2 * math.pi)).numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(delta_mhz, u00.real.numpy(), label="Re(u_00)", linewidth=1.4)
    ax.plot(delta_mhz, u01.imag.numpy(), label="Im(u_01)", linewidth=1.4)

    alpha_np = alpha_t.detach().cpu().numpy().reshape(-1)
    for i, d_mhz in enumerate(model.delta_centers_mhz):
        ai = float(alpha_np[i])
        lo = d_mhz - model.robustness_window_mhz
        hi = d_mhz + model.robustness_window_mhz
        ax.hlines(
            y=math.cos(ai / 2), xmin=lo, xmax=hi,
            colors="red", linestyles="dashed", linewidth=1.8,
            label="target Re(u_00)" if i == 0 else None,
        )
        ax.hlines(
            y=-math.sin(ai / 2), xmin=lo, xmax=hi,
            colors="green", linestyles="dashed", linewidth=1.8,
            label="target Im(u_01)" if i == 0 else None,
        )
        ax.axvspan(lo, hi, color="gray", alpha=0.12)

    fidelity_val = compute_fidelity(model, alpha, sample_size=sample_size_for_fidelity)
    runtime_us = compute_runtime(model, alpha)

    ax.set_xlabel("Detuning $\\delta$ (MHz)")
    ax.set_ylabel("Matrix element")
    ax.set_ylim(-1.25, 1.25)
    ax.set_title(
        rf"$\Omega$={model.Omega_mhz:.0f} MHz, $K$={model.K}   "
        rf"$\alpha/\pi$ = [" +
        ", ".join(f"{a/math.pi:.2f}" for a in alpha_np) + "]   "
        f"F={fidelity_val:.4f}, T={runtime_us:.3f} μs"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return fidelity_val


# ─────────────────────────────────────────────────────────────────────────────
#  Scaling-law figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_scaling_law(summary_df: pd.DataFrame, out_path: str):
    """Avg and min fidelity vs K, one line per Omega; log-infidelity twin axis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric, title in zip(
        axes, ["avg_fidelity", "min_fidelity"], ["Average fidelity", "Minimum fidelity"]
    ):
        for omega, sub in summary_df.groupby("Omega_mhz"):
            sub = sub.sort_values("K")
            ax.plot(sub["K"], sub[metric], "o-", label=f"$\\Omega$={omega:.0f} MHz")
        ax.set_xlabel("K")
        ax.set_ylabel(title)
        ax.set_title(title + " vs. K")
        ax.grid(alpha=0.3)
        ax.legend()

        ax2 = ax.twinx()
        for omega, sub in summary_df.groupby("Omega_mhz"):
            sub = sub.sort_values("K")
            infid = (1.0 - sub[metric]).clip(lower=1e-8)
            ax2.plot(sub["K"], infid, "o--", alpha=0.35, color="gray")
        ax2.set_yscale("log")
        ax2.set_ylabel("1 − fidelity (log)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_runtime_vs_fidelity(full_df: pd.DataFrame, out_path: str):
    """Scatter of runtime vs fidelity; twin panel with log infidelity."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    markers = {50: "o", 70: "s", 100: "^", 160: "D"}
    cmap = plt.get_cmap("viridis")
    omegas = sorted(full_df["Omega_mhz"].unique())
    colors = {o: cmap(i / max(1, len(omegas) - 1)) for i, o in enumerate(omegas)}

    for ax, use_log in zip(axes, [False, True]):
        for (omega, K), sub in full_df.groupby(["Omega_mhz", "K"]):
            y = sub["fidelity"] if not use_log else (1 - sub["fidelity"]).clip(lower=1e-8)
            ax.scatter(
                sub["runtime_us"], y,
                c=[colors.get(omega, "k")],
                marker=markers.get(int(K), "x"),
                s=18, alpha=0.5,
                label=f"Ω={omega:.0f}, K={int(K)}",
            )
        ax.set_xlabel("Total pulse runtime (μs)")
        if use_log:
            ax.set_ylabel("1 − fidelity")
            ax.set_yscale("log")
            ax.set_title("Log infidelity")
        else:
            ax.set_ylabel("Fidelity")
            ax.set_title("Fidelity")
        ax.grid(alpha=0.3)

    # Single shared legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate while preserving order
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if uniq:
        fig.legend(
            [h for h, _ in uniq], [l for _, l in uniq],
            loc="upper center", bbox_to_anchor=(0.5, 1.02),
            ncol=min(6, len(uniq)), fontsize=9,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  PCA figures (ported, adapted for PulseNet signature)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_fourier_fit(coeffs, n_fourier, alpha_grid):
    design = [np.ones(len(alpha_grid))]
    for k in range(1, n_fourier + 1):
        design.append(np.cos(k * alpha_grid / 2))
        design.append(np.sin(k * alpha_grid / 2))
    return np.column_stack(design) @ coeffs


def _evaluate_poly_fit(coeffs, alpha_grid):
    return np.polyval(coeffs, alpha_grid)


def _evaluate_fit(fit, fit_type, alpha_grid):
    if fit_type == "fourier_fit":
        return _evaluate_fourier_fit(fit["coeffs"], fit["n_fourier"], alpha_grid)
    _, coeffs, _ = fit
    return _evaluate_poly_fit(coeffs, alpha_grid)


def plot_pca_overview(pca_result, peak_index, out_path, fit_type="fourier_fit"):
    """4-panel PCA overview: SV spectrum, residual variance, amplitudes, phi samples."""
    alpha_grid = pca_result["alpha_grid"]
    phi_matrix = pca_result["phi_matrix"]
    S = pca_result["singular_values"]
    er = pca_result["explained_variance_ratio"]
    cumulative = np.cumsum(er)
    n_dof = pca_result["n_effective_dof"]
    amplitudes = pca_result["amplitudes"]
    fits = pca_result["amplitude_fits"] if fit_type == "fourier_fit" else pca_result["polyfit_fits"]

    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polynomial"
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax = axes[0, 0]
    ax.semilogy(S / S[0], "o-", markersize=4)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Normalized singular value")
    ax.set_title(f"SV spectrum (peak {peak_index})")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    residual = np.clip(1.0 - cumulative, a_min=1e-16, a_max=None)
    ax.semilogy(residual, "o-", markersize=4)
    ax.axvline(x=n_dof - 1, color="g", linestyle="--", alpha=0.6, label=f"d={n_dof}")
    ax.set_xlabel("# components")
    ax.set_ylabel("1 − cumulative explained variance")
    ax.set_title("Residual variance")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for j in range(min(n_dof, 5)):
        ax.plot(alpha_grid / math.pi, amplitudes[:, j], label=f"$A_{j}$", linewidth=1.2)
        fitted = _evaluate_fit(fits[j], fit_type, alpha_grid)
        ax.plot(alpha_grid / math.pi, fitted, "--", alpha=0.6, linewidth=1.0)
    ax.set_xlabel(r"$\alpha / \pi$")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Amplitudes (solid=data, dashed={fit_label} fit)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    n_show = min(5, phi_matrix.shape[0])
    idxs = np.linspace(0, phi_matrix.shape[0] - 1, n_show, dtype=int)
    for idx in idxs:
        ax.plot(phi_matrix[idx], label=f"α={alpha_grid[idx] / math.pi:.2f}π", alpha=0.75)
    ax.set_xlabel("Phase index j")
    ax.set_ylabel("$\\phi_j$")
    ax.set_title("Sample φ trajectories")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f"PCA overview (peak {peak_index}, {fit_label})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_amplitude_components(pca_result, peak_index, out_path, fit_type="fourier_fit"):
    """Per-component amplitude fit subplots."""
    alpha_grid = pca_result["alpha_grid"]
    amplitudes = pca_result["amplitudes"]
    n_dof = pca_result["n_effective_dof"]
    fits = pca_result["amplitude_fits"] if fit_type == "fourier_fit" else pca_result["polyfit_fits"]
    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polynomial"

    n_cols = min(3, n_dof)
    n_rows = math.ceil(n_dof / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    alpha_fine = np.linspace(alpha_grid.min(), alpha_grid.max(), 500)
    for k in range(n_dof):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        ax.plot(alpha_grid / math.pi, amplitudes[:, k], ".", color="C0",
                markersize=3, alpha=0.5, label="data")
        ax.plot(alpha_fine / math.pi, _evaluate_fit(fits[k], fit_type, alpha_fine),
                "-", color="C1", linewidth=1.6, label=f"{fit_label} fit")
        ax.set_xlabel(r"$\alpha\,/\,\pi$")
        ax.set_ylabel(f"$A_{{{k}}}(\\alpha)$")
        if fit_type == "fourier_fit":
            ax.set_title(f"k={k}, n_fourier={fits[k]['n_fourier']}, res={fits[k]['residual']:.2e}")
        else:
            deg, _, r2 = fits[k]
            ax.set_title(f"k={k}, deg={deg}, $R^2$={r2:.4f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    for k in range(n_dof, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].set_visible(False)
    plt.suptitle(f"{fit_label} fit of $A_k(\\alpha)$ — peak {peak_index}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _make_peak_input(alpha_scalar, peak_index, N_peaks, device=None):
    x = torch.zeros(N_peaks, dtype=torch.float64, device=device)
    x[peak_index] = float(alpha_scalar)
    return x


def plot_comparative_pulses(
    model: PulseNet,
    pca_result,
    peak_index: int,
    out_path: str,
    alpha_vals=None,
    fit_type: str = "fourier_fit",
    n_basis: Optional[int] = None,
):
    """Side-by-side (Omega_x(t), Omega_z(t)) from NN vs PCA-reconstruction at selected alphas."""
    from .pca_analysis import reconstruct_phi_analytical, reconstruct_phi_polyfit

    if alpha_vals is None:
        alpha_vals = [math.pi / 4, math.pi / 2, math.pi, 2 * math.pi, 3 * math.pi]

    reconstruct = (
        reconstruct_phi_analytical if fit_type == "fourier_fit"
        else reconstruct_phi_polyfit
    )
    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polyfit"
    if n_basis is not None:
        fit_label += f" (d={n_basis})"

    n_alpha = len(alpha_vals)
    fig, axes = plt.subplots(n_alpha, 2, figsize=(14, 3.2 * n_alpha), squeeze=False)
    model.eval()
    dev = next(model.parameters()).device

    for row, alpha in enumerate(alpha_vals):
        with torch.no_grad():
            x = _make_peak_input(alpha, peak_index, model.N_peaks, device=dev).unsqueeze(0)
            phi_nn = model(x).squeeze(0).detach().cpu()
        pdf_nn = phi_to_pulse_df(phi_nn, model.Omega_mhz, model.Delta_0_mhz)
        phi_fit = reconstruct(pca_result, alpha, n_basis=n_basis)
        pdf_fit = phi_to_pulse_df(phi_fit, model.Omega_mhz, model.Delta_0_mhz)

        t_nn = np.cumsum(np.r_[0.0, pdf_nn["t (us)"].to_numpy()])
        t_ft = np.cumsum(np.r_[0.0, pdf_fit["t (us)"].to_numpy()])
        for col, field in enumerate(["Omega_x (2pi MHz)", "Omega_z (2pi MHz)"]):
            ax = axes[row, col]
            h_nn = pdf_nn[field].to_numpy()
            h_ft = pdf_fit[field].to_numpy()
            ax.step(t_nn, np.r_[h_nn, h_nn[-1]], where="post", label="NN", linewidth=1.4)
            ax.step(t_ft, np.r_[h_ft, h_ft[-1]], where="post",
                    linestyle="--", alpha=0.85, label=fit_label, linewidth=1.4)
            ax.set_ylabel(field)
            ax.set_title(rf"$\alpha={alpha/math.pi:.2f}\pi$   {field}")
            ax.grid(alpha=0.3)
            if row == n_alpha - 1:
                ax.set_xlabel("Time (μs)")
            ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparative_matrix_elements(
    model: PulseNet,
    pca_result,
    peak_index: int,
    out_path: str,
    alpha_vals=None,
    fit_type: str = "fourier_fit",
    n_basis: Optional[int] = None,
    n_delta: int = 1024,
    sample_size_for_fidelity: int = 2000,
):
    """Matrix element vs detuning: NN vs PCA reconstruction.  Returns fidelity dict."""
    from .pca_analysis import reconstruct_phi_analytical, reconstruct_phi_polyfit
    from .physics import fidelity_from_pulse

    if alpha_vals is None:
        alpha_vals = [math.pi / 4, math.pi / 2, math.pi, 2 * math.pi]

    reconstruct = (
        reconstruct_phi_analytical if fit_type == "fourier_fit"
        else reconstruct_phi_polyfit
    )
    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polyfit"
    if n_basis is not None:
        fit_label += f" (d={n_basis})"

    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    Delta_0_ang = model.Delta_0_ang
    Omega_ang = model.Omega_ang
    delta_range = torch.linspace(-Delta_0_ang, Delta_0_ang, n_delta, dtype=torch.float64)
    delta_mhz = (delta_range / (2 * math.pi)).numpy()

    def u_profile(phi):
        if isinstance(phi, np.ndarray):
            phi_t = torch.tensor(phi, dtype=torch.float64)
        else:
            phi_t = phi.detach().to(torch.float64)
        U = build_qsp_unitary(phi_t, delta_range, Delta_0_ang, Omega_ang)
        U = H @ U @ H
        return U[:, 0, 0].detach().cpu().numpy(), U[:, 0, 1].detach().cpu().numpy()

    dev = next(model.parameters()).device
    n_alpha = len(alpha_vals)
    fig, axes = plt.subplots(n_alpha, 1, figsize=(12, 4 * n_alpha), squeeze=False)

    fidelity_results = {}
    for row, alpha in enumerate(alpha_vals):
        with torch.no_grad():
            x = _make_peak_input(alpha, peak_index, model.N_peaks, device=dev).unsqueeze(0)
            phi_nn = model(x).squeeze(0).detach().cpu()
        phi_fit = reconstruct(pca_result, alpha, n_basis=n_basis)

        u00_nn, u01_nn = u_profile(phi_nn)
        u00_ft, u01_ft = u_profile(phi_fit)

        delta_targets = np.asarray(model.delta_centers_mhz, dtype=float)
        alpha_targets = np.zeros(len(delta_targets))
        alpha_targets[peak_index] = alpha
        pdf_nn = phi_to_pulse_df(phi_nn, model.Omega_mhz, model.Delta_0_mhz)
        pdf_ft = phi_to_pulse_df(phi_fit, model.Omega_mhz, model.Delta_0_mhz)
        fid_nn = fidelity_from_pulse(
            pdf_nn, delta_targets, alpha_targets,
            robustness_window_mhz=model.robustness_window_mhz,
            sample_size=sample_size_for_fidelity,
        )
        fid_ft = fidelity_from_pulse(
            pdf_ft, delta_targets, alpha_targets,
            robustness_window_mhz=model.robustness_window_mhz,
            sample_size=sample_size_for_fidelity,
        )
        fidelity_results[float(alpha)] = {"fidelity_nn": fid_nn, "fidelity_fit": fid_ft}

        ax = axes[row, 0]
        ax.plot(delta_mhz, u00_nn.real, color="C0", linewidth=1.3, linestyle="--",
                label="Re(u00) NN")
        ax.plot(delta_mhz, u01_nn.imag, color="C1", linewidth=1.3, linestyle="--",
                label="Im(u01) NN")
        ax.plot(delta_mhz, u00_ft.real, color="C2", linewidth=1.3, alpha=0.8,
                label=f"Re(u00) {fit_label}")
        ax.plot(delta_mhz, u01_ft.imag, color="C3", linewidth=1.3, alpha=0.8,
                label=f"Im(u01) {fit_label}")
        for j, dc in enumerate(model.delta_centers_mhz):
            tgt_alpha = alpha if j == peak_index else 0.0
            ax.hlines(y=math.cos(tgt_alpha / 2),
                      xmin=dc - model.robustness_window_mhz,
                      xmax=dc + model.robustness_window_mhz,
                      colors="red", linestyles="dotted", linewidth=1.5)
            ax.hlines(y=-math.sin(tgt_alpha / 2),
                      xmin=dc - model.robustness_window_mhz,
                      xmax=dc + model.robustness_window_mhz,
                      colors="green", linestyles="dotted", linewidth=1.5)
            ax.axvspan(dc - model.robustness_window_mhz,
                       dc + model.robustness_window_mhz, color="gray", alpha=0.1)
        ax.set_ylim(-1.3, 1.3)
        ax.set_title(
            rf"peak {peak_index}, $\alpha$={alpha/math.pi:.2f}π    "
            f"F_NN={fid_nn:.4f}, F_{fit_label}={fid_ft:.4f}"
        )
        ax.set_ylabel("Matrix element")
        if row == n_alpha - 1:
            ax.set_xlabel(r"Detuning $\delta$ (MHz)")
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return fidelity_results
