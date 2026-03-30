"""PCA analysis, reconstruction, and analytical form output."""

import math
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .model import PulseGeneratorNet
from .evaluation import verify_phi_fidelities


# ─────────────────────────────────────────────────────────────────────────────
#  Amplitude fitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_fourier(alpha_np, amplitudes, n_dof):
    """Fit each amplitude A_j(alpha) with a Fourier series in alpha/2."""
    N = len(alpha_np)
    fits = []
    for j in range(n_dof):
        amp_j = amplitudes[:, j]
        n_fourier = min(5, N // 4)
        design = [np.ones(N)]
        for k in range(1, n_fourier + 1):
            design.append(np.cos(k * alpha_np / 2))
            design.append(np.sin(k * alpha_np / 2))
        design = np.column_stack(design)
        coeffs, _, _, _ = np.linalg.lstsq(design, amp_j, rcond=None)
        residual = float(np.mean((amp_j - design @ coeffs) ** 2))
        fits.append({
            "coeffs": coeffs,
            "n_fourier": n_fourier,
            "residual": residual,
        })
    return fits


def _fit_polynomial(alpha_np, amplitudes, n_dof, max_deg=10):
    """Fit each amplitude A_k(alpha) with the best-degree polynomial (selected by AIC)."""
    M = len(alpha_np)
    fits = []
    for k in range(n_dof):
        amp_k = amplitudes[:, k]
        best_aic = np.inf
        best_deg = 1
        best_coeffs = None

        for deg in range(1, max_deg + 1):
            coeffs = np.polyfit(alpha_np, amp_k, deg)
            rss = float(np.sum((amp_k - np.polyval(coeffs, alpha_np)) ** 2))
            aic = M * np.log(rss / M + 1e-30) + 2 * (deg + 1)
            if aic < best_aic:
                best_aic = aic
                best_deg = deg
                best_coeffs = coeffs

        fitted = np.polyval(best_coeffs, alpha_np)
        rss = float(np.sum((amp_k - fitted) ** 2))
        ss_tot = float(np.sum((amp_k - amp_k.mean()) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
        fits.append((best_deg, best_coeffs, r2))
    return fits


# ─────────────────────────────────────────────────────────────────────────────
#  Main PCA analysis
# ─────────────────────────────────────────────────────────────────────────────

def pca_analysis(
    net: PulseGeneratorNet,
    M: int = 200,
    explained_var_threshold: float = 0.999,
    out_dir: str = "nn_pulse_output",
    alpha_max: float = 4 * math.pi,
    polyfit_max_deg: int = 10,
) -> Dict:
    """Perform PCA on the phi vectors generated across alpha in [0, alpha_max].

    For each fit type (fourier_fit, polyfit), generates the full pipeline:
      - pca_overview (4-panel plot)
      - pca_components (2D projection)
      - amplitude_components (per-component fit subplots)
      - basis_functions and coefficients CSVs

    Returns
    -------
    Dict with keys: "alpha_grid", "phi_matrix", "mean_phi",
    "principal_components", "singular_values", "explained_variance_ratio",
    "n_effective_dof", "amplitudes", "amplitude_fits", "polyfit_fits", "fidelities"
    """
    net.eval()
    alpha_grid = torch.linspace(0, alpha_max, 2 * M, dtype=torch.float64)

    with torch.no_grad():
        phi_matrix = net(alpha_grid).cpu().numpy()  # (2*M, K+1)

    mean_phi = phi_matrix.mean(axis=0)
    phi_centered = phi_matrix - mean_phi

    _, S, Vt = np.linalg.svd(phi_centered, full_matrices=False)

    var = S ** 2
    total_var = var.sum()
    explained_ratio = var / total_var if total_var > 0 else var
    cumulative = np.cumsum(explained_ratio)

    n_dof = int(np.searchsorted(cumulative, explained_var_threshold) + 1)
    n_dof = min(n_dof, len(S))

    # Principal components: (K+1, n_dof)
    pc = Vt[:n_dof, :].T

    # Amplitudes: projection onto PCs
    amplitudes = phi_centered @ pc  # (2*M, n_dof)

    # Fit amplitude functions with both methods
    alpha_np = alpha_grid.numpy()
    N = len(alpha_np)
    amplitude_fits = _fit_fourier(alpha_np, amplitudes, n_dof)
    polyfit_fits = _fit_polynomial(alpha_np, amplitudes, n_dof, polyfit_max_deg)

    # Verify fidelity of all 2*M phi vectors using independent calculation
    fidelities = verify_phi_fidelities(phi_matrix, alpha_np, net)
    min_fid = float(fidelities.min())
    mean_fid = float(fidelities.mean())
    median_fid = float(np.median(fidelities))
    print(f"  Fidelity (independent): min={min_fid:.6f}, "
          f"mean={mean_fid:.6f}, median={median_fid:.6f}")
    n_below = int(np.sum(fidelities < 0.96))
    if n_below > 0:
        print(f"  WARNING: {n_below}/{N} phi vectors have fidelity < 96%")
    else:
        print(f"  All {N} phi vectors have fidelity >= 96%")

    os.makedirs(out_dir, exist_ok=True)

    result = {
        "alpha_grid": alpha_np,
        "phi_matrix": phi_matrix,
        "mean_phi": mean_phi,
        "principal_components": pc,
        "singular_values": S,
        "explained_variance_ratio": explained_ratio,
        "n_effective_dof": n_dof,
        "amplitudes": amplitudes,
        "amplitude_fits": amplitude_fits,
        "polyfit_fits": polyfit_fits,
        "fidelities": fidelities,
    }

    # Deferred imports to break pca <-> plotting circular dependency
    from .plotting import plot_pca_overview, plot_pca_components, plot_amplitude_components

    for fit_type in ["fourier_fit", "polyfit"]:
        fit_dir = os.path.join(out_dir, fit_type)
        fits = amplitude_fits if fit_type == "fourier_fit" else polyfit_fits

        plot_pca_overview(
            alpha_np, phi_matrix, S, explained_ratio, cumulative,
            n_dof, amplitudes, fits, fit_type=fit_type,
            peak_index=net.peak_index, out_dir=fit_dir,
            explained_var_threshold=explained_var_threshold,
        )
        plot_pca_components(amplitudes, alpha_np, net.peak_index, fit_dir)
        plot_amplitude_components(
            alpha_np, amplitudes, fits, fit_type=fit_type,
            n_dof=n_dof, peak_index=net.peak_index, out_dir=fit_dir,
        )
        save_pca_csv(result, net.peak_index, fit_dir, fit_type)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_pca_csv(pca_result: Dict, peak_index: int, out_dir: str, fit_type: str):
    """Save basis functions and fit coefficients to out_dir/{basis_functions,coefficients}/."""
    basis_dir = os.path.join(out_dir, "basis_functions")
    coeffs_dir = os.path.join(out_dir, "coefficients")
    os.makedirs(basis_dir, exist_ok=True)
    os.makedirs(coeffs_dir, exist_ok=True)

    pc = pca_result["principal_components"]  # (K+1, n_dof)
    n_dof = pca_result["n_effective_dof"]
    mean_phi = pca_result["mean_phi"]

    # --- Basis functions CSV (same for both fit types) ---
    K_plus_1 = pc.shape[0]
    basis_data = {"phase_index": np.arange(K_plus_1), "mean_phi": mean_phi}
    for j in range(n_dof):
        basis_data[f"b_{j}"] = pc[:, j]
    pd.DataFrame(basis_data).to_csv(
        os.path.join(basis_dir, f"basis_functions_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )

    # --- Coefficients CSV (depends on fit type) ---
    if fit_type == "fourier_fit":
        fits = pca_result["amplitude_fits"]
        rows = []
        for j in range(n_dof):
            coeffs = fits[j]["coeffs"]
            n_fourier = fits[j]["n_fourier"]
            row = {"component": j, "a_0": coeffs[0]}
            for k in range(1, n_fourier + 1):
                row[f"a_cos{k}"] = coeffs[2 * k - 1]
                row[f"a_sin{k}"] = coeffs[2 * k]
            row["residual"] = fits[j]["residual"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(coeffs_dir, f"fourier_coefficients_peak{peak_index}.csv"),
            index=False, float_format="%.8f",
        )
    else:
        fits = pca_result["polyfit_fits"]
        max_deg = max(deg for deg, _, _ in fits)
        alpha_grid = pca_result["alpha_grid"]
        rows = []
        for k in range(n_dof):
            deg, coeffs, r2 = fits[k]
            coeffs_asc = coeffs[::-1]
            row = {"component": k, "degree": deg, "R2": round(r2, 8)}
            for i in range(max_deg + 1):
                row[f"c_{i}"] = float(coeffs_asc[i]) if i <= deg else 0.0
            amp_k = pca_result["amplitudes"][:, k]
            rss = float(np.sum((amp_k - np.polyval(coeffs, alpha_grid)) ** 2))
            row["residual_mse"] = rss / len(alpha_grid)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(coeffs_dir, f"polyfit_coefficients_peak{peak_index}.csv"),
            index=False, float_format="%.8f",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_phi_analytical(pca_result: Dict, alpha: float, n_basis: int = None) -> np.ndarray:
    """Reconstruct phi(alpha) from the PCA Fourier decomposition.

    phi(alpha) = mean_phi + sum_j A_j(alpha) * pc_j

    Parameters
    ----------
    n_basis : int or None
        Number of basis vectors to use. None uses all effective DOF.
    """
    mean_phi = pca_result["mean_phi"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    fits = pca_result["amplitude_fits"]
    n_dof = pca_result["n_effective_dof"]
    if n_basis is None:
        n_basis = n_dof
    n_basis = min(n_basis, n_dof)

    phi_recon = mean_phi.copy()
    for j in range(n_basis):
        coeffs = fits[j]["coeffs"]
        n_fourier = fits[j]["n_fourier"]
        val = coeffs[0]
        for k in range(1, n_fourier + 1):
            val += coeffs[2 * k - 1] * np.cos(k * alpha/2)
            val += coeffs[2 * k] * np.sin(k * alpha/2)
        phi_recon += val * pc[:, j]

    return phi_recon


def reconstruct_phi_polyfit(pca_result: Dict, alpha: float, n_basis: int = None) -> np.ndarray:
    """Reconstruct phi(alpha) from the PCA polynomial fit decomposition.

    phi(alpha) = mean_phi + sum_j A_j(alpha) * pc_j

    where A_j(alpha) is evaluated from its polynomial fit coefficients.

    Parameters
    ----------
    n_basis : int or None
        Number of basis vectors to use. None uses all effective DOF.
    """
    mean_phi = pca_result["mean_phi"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    fits = pca_result["polyfit_fits"]  # list of (deg, coeffs_descending, r2)
    n_dof = pca_result["n_effective_dof"]
    if n_basis is None:
        n_basis = n_dof
    n_basis = min(n_basis, n_dof)

    phi_recon = mean_phi.copy()
    for j in range(n_basis):
        _deg, coeffs, _r2 = fits[j]
        val = float(np.polyval(coeffs, alpha))
        phi_recon += val * pc[:, j]

    return phi_recon


# ─────────────────────────────────────────────────────────────────────────────
#  Analytical form output
# ─────────────────────────────────────────────────────────────────────────────

def format_analytical_form(
    pca_result: Dict,
    fit_type: str = "fourier_fit",
    coeff_threshold: float = 1e-3,
    fidelity_results: Dict = None,
) -> str:
    """Format the PCA decomposition as a human-readable analytical expression.

    Parameters
    ----------
    fit_type : 'fourier_fit' or 'polyfit'
        Which fit to format.
    fidelity_results : dict, optional
        Mapping theta -> {"fidelity_nn": float, "fidelity_pca": float}.
        If provided, fidelity comparison is appended.

    Returns a multi-line string describing:
      H_c^i(t; theta) = phi_mean(t) + sum_j A_j(theta) * b_j^i(t)
    with A_j as Fourier series or polynomial and b_j as weight vectors over phase indices.
    """
    n_dof = pca_result["n_effective_dof"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    mean_phi = pca_result["mean_phi"]
    sv = pca_result["singular_values"]
    ev = pca_result["explained_variance_ratio"]

    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polynomial"

    K = pc.shape[0] - 1
    lines = []
    lines.append("=" * 70)
    lines.append(f"ANALYTICAL DECOMPOSITION ({fit_label} Fit)")
    lines.append(
        f"  H_c^i(t; alpha) = phi_mean(t)"
        f" + sum_{{j=0}}^{{{n_dof-1}}} A_j(alpha) * b_j(t)")
    lines.append(f"  Effective linear components: d = {n_dof}  (K = {K})")
    lines.append(f"  Explained variance: {sum(ev[:n_dof])*100:.2f}%")
    lines.append("=" * 70)

    nonzero_mean = np.where(np.abs(mean_phi) > coeff_threshold)[0]
    lines.append(f"\nphi_mean: {len(nonzero_mean)} / {K+1} nonzero entries")
    lines.append(f"  ||phi_mean|| = {np.linalg.norm(mean_phi):.4f}")

    for j in range(n_dof):
        lines.append(f"\n{'─' * 50}")
        lines.append(
            f"Component j = {j}  (singular value = {sv[j]:.4f}, "
            f"variance = {ev[j]*100:.2f}%)")
        lines.append("─" * 50)

        if fit_type == "fourier_fit":
            fits = pca_result["amplitude_fits"]
            coeffs = fits[j]["coeffs"]
            n_fourier = fits[j]["n_fourier"]
            residual = fits[j]["residual"]

            terms = []
            if abs(coeffs[0]) > coeff_threshold:
                terms.append(f"{coeffs[0]:+.4f}")
            for k in range(1, n_fourier + 1):
                c_cos = coeffs[2 * k - 1]
                c_sin = coeffs[2 * k]
                if abs(c_cos) > coeff_threshold:
                    terms.append(f"{c_cos:+.4f} cos({k}*alpha)")
                if abs(c_sin) > coeff_threshold:
                    terms.append(f"{c_sin:+.4f} sin({k}*alpha)")

            if not terms:
                a_str = "0"
            else:
                a_str = " ".join(terms)
                if a_str.startswith("+"):
                    a_str = a_str[1:]

            lines.append(f"  A_{j}(alpha) = {a_str}")
            lines.append(f"  Fourier fit residual: {residual:.2e}")
        else:
            fits = pca_result["polyfit_fits"]
            deg, coeffs_desc, r2 = fits[j]
            coeffs_asc = coeffs_desc[::-1]

            terms = []
            for i in range(deg + 1):
                c = coeffs_asc[i]
                if abs(c) > coeff_threshold:
                    if i == 0:
                        terms.append(f"{c:+.4f}")
                    elif i == 1:
                        terms.append(f"{c:+.4f}*alpha")
                    else:
                        terms.append(f"{c:+.4f}*alpha^{i}")

            if not terms:
                a_str = "0"
            else:
                a_str = " ".join(terms)
                if a_str.startswith("+"):
                    a_str = a_str[1:]

            lines.append(f"  A_{j}(alpha) = {a_str}")
            lines.append(f"  Polynomial fit: deg={deg}, R^2={r2:.6f}")

        bj = pc[:, j]
        sorted_idx = np.argsort(np.abs(bj))[::-1]
        top_n = min(8, len(sorted_idx))

        lines.append(f"  b_{j}(t): dominant phase indices (top {top_n}):")
        for idx in sorted_idx[:top_n]:
            if abs(bj[idx]) < coeff_threshold:
                break
            lines.append(f"    phi[{idx:3d}] weight = {bj[idx]:+.4f}")

        lines.append(f"  ||b_{j}|| = {np.linalg.norm(bj):.4f}")

    lines.append("\n" + "=" * 70)
    lines.append("RECONSTRUCTION FORMULA")
    lines.append("  For a given alpha, the QSP phases are:")
    lines.append(
        f"    phi_j = mean_phi_j"
        f" + sum_{{k=0}}^{{{n_dof-1}}} A_k(alpha) * b_k[j]")
    lines.append("  Then the pulse schedule is built via phi_to_pulse_df(phi).")
    lines.append("=" * 70)

    if fidelity_results:
        lines.append("\n" + "=" * 70)
        lines.append("FIDELITY COMPARISON: NN (original) vs PCA (reconstructed)")
        lines.append("=" * 70)
        for alpha in sorted(fidelity_results.keys()):
            fid = fidelity_results[alpha]
            alpha_label = f"{alpha/math.pi:.2f}*pi"
            lines.append(
                f"  alpha = {alpha_label:>10s}  |  "
                f"NN = {fid['fidelity_nn']:.6f}  |  "
                f"Fit = {fid['fidelity_fit']:.6f}")
        lines.append("=" * 70)

    return "\n".join(lines)
