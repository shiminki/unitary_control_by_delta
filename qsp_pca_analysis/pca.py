"""PCA analysis, reconstruction, and analytical form output."""

import math
import os
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .model import PulseGeneratorNet
from .evaluation import verify_phi_fidelities


def pca_analysis(
    net: PulseGeneratorNet,
    M: int = 200,
    explained_var_threshold: float = 0.999,
    out_dir: str = "nn_pulse_output",
) -> Dict:
    """Perform PCA on the phi vectors generated across theta in [0, 2pi].

    Parameters
    ----------
    net : trained PulseGeneratorNet
    M : number of theta samples
    explained_var_threshold : cumulative variance threshold for selecting components

    Returns
    -------
    Dict with keys: "theta_grid", "phi_matrix", "mean_phi",
    "principal_components", "singular_values", "explained_variance_ratio",
    "n_effective_dof", "amplitudes", "amplitude_fits", "fidelities"
    """
    net.eval()
    theta_grid = torch.linspace(0, 2 * math.pi, M, dtype=torch.float64)

    with torch.no_grad():
        phi_matrix = net(theta_grid).cpu().numpy()  # (M, K+1)

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
    amplitudes = phi_centered @ pc  # (M, n_dof)

    # Fit each A_j(theta) with a Fourier series
    theta_np = theta_grid.numpy()
    amplitude_fits = []
    for j in range(n_dof):
        amp_j = amplitudes[:, j]
        n_fourier = min(10, M // 4)
        design = [np.ones(M)]
        for k in range(1, n_fourier + 1):
            design.append(np.cos(k * theta_np))
            design.append(np.sin(k * theta_np))
        design = np.column_stack(design)
        coeffs, _, _, _ = np.linalg.lstsq(design, amp_j, rcond=None)
        residual = float(np.mean((amp_j - design @ coeffs) ** 2))
        amplitude_fits.append({
            "coeffs": coeffs,
            "n_fourier": n_fourier,
            "residual": residual,
        })

    # Verify fidelity of all M phi vectors using independent calculation
    fidelities = verify_phi_fidelities(phi_matrix, theta_np, net)
    min_fid = float(fidelities.min())
    mean_fid = float(fidelities.mean())
    median_fid = float(np.median(fidelities))
    print(f"  Fidelity (independent): min={min_fid:.6f}, "
          f"mean={mean_fid:.6f}, median={median_fid:.6f}")
    n_below = int(np.sum(fidelities < 0.96))
    if n_below > 0:
        print(f"  WARNING: {n_below}/{M} phi vectors have fidelity < 96%")
    else:
        print(f"  All {M} phi vectors have fidelity >= 96%")

    os.makedirs(out_dir, exist_ok=True)

    # Deferred imports to break pca <-> plotting circular dependency
    from .plotting import plot_pca_results, plot_tsne

    plot_pca_results(
        theta_np, phi_matrix, S, explained_ratio, cumulative,
        n_dof, amplitudes, amplitude_fits, peak_index=net.peak_index,
        out_dir=out_dir, explained_var_threshold=explained_var_threshold,
    )
    plot_tsne(phi_matrix, theta_np, net.peak_index, out_dir)

    result = {
        "theta_grid": theta_np,
        "phi_matrix": phi_matrix,
        "mean_phi": mean_phi,
        "principal_components": pc,
        "singular_values": S,
        "explained_variance_ratio": explained_ratio,
        "n_effective_dof": n_dof,
        "amplitudes": amplitudes,
        "amplitude_fits": amplitude_fits,
        "fidelities": fidelities,
    }

    save_pca_csv(result, net.peak_index, out_dir)

    return result


def save_pca_csv(pca_result: Dict, peak_index: int, out_dir: str):
    """Save basis functions b_j^i(t) and Fourier coefficients of A_j(theta) as CSV files."""
    os.makedirs(out_dir, exist_ok=True)
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    n_dof = pca_result["n_effective_dof"]
    mean_phi = pca_result["mean_phi"]
    fits = pca_result["amplitude_fits"]

    # --- Basis functions CSV ---
    K_plus_1 = pc.shape[0]
    basis_data = {"phase_index": np.arange(K_plus_1), "mean_phi": mean_phi}
    for j in range(n_dof):
        basis_data[f"b_{j}"] = pc[:, j]
    df_basis = pd.DataFrame(basis_data)
    df_basis.to_csv(
        os.path.join(out_dir, f"basis_functions_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )

    # --- Fourier coefficients CSV ---
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
    df_coeffs = pd.DataFrame(rows)
    df_coeffs.to_csv(
        os.path.join(out_dir, f"fourier_coefficients_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )


def reconstruct_phi_analytical(pca_result: Dict, theta: float) -> np.ndarray:
    """Reconstruct phi(theta) from the PCA decomposition.

    phi(theta) = mean_phi + sum_j A_j(theta) * pc_j

    where A_j(theta) is evaluated from its Fourier fit coefficients.
    """
    mean_phi = pca_result["mean_phi"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    fits = pca_result["amplitude_fits"]
    n_dof = pca_result["n_effective_dof"]

    phi_recon = mean_phi.copy()
    for j in range(n_dof):
        coeffs = fits[j]["coeffs"]
        n_fourier = fits[j]["n_fourier"]
        val = coeffs[0]
        for k in range(1, n_fourier + 1):
            val += coeffs[2 * k - 1] * np.cos(k * theta)
            val += coeffs[2 * k] * np.sin(k * theta)
        phi_recon += val * pc[:, j]

    return phi_recon


def format_analytical_form(
    pca_result: Dict,
    coeff_threshold: float = 1e-3,
    fidelity_results: Dict = None,
) -> str:
    """Format the PCA decomposition as a human-readable analytical expression.

    Parameters
    ----------
    fidelity_results : dict, optional
        Mapping theta -> {"fidelity_nn": float, "fidelity_pca": float}.
        If provided, fidelity comparison is appended.

    Returns a multi-line string describing:
      H_c^i(t; theta) = phi_mean(t) + sum_j A_j(theta) * b_j^i(t)
    with A_j as Fourier series and b_j as weight vectors over phase indices.
    """
    n_dof = pca_result["n_effective_dof"]
    pc = pca_result["principal_components"]  # (K+1, n_dof)
    mean_phi = pca_result["mean_phi"]
    fits = pca_result["amplitude_fits"]
    sv = pca_result["singular_values"]
    ev = pca_result["explained_variance_ratio"]

    K = pc.shape[0] - 1
    lines = []
    lines.append("=" * 70)
    lines.append("ANALYTICAL DECOMPOSITION")
    lines.append(
        f"  H_c^i(t; theta) = phi_mean(t)"
        f" + sum_{{j=0}}^{{{n_dof-1}}} A_j(theta) * b_j(t)")
    lines.append(f"  Effective DOF: d = {n_dof}  (K = {K})")
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
                terms.append(f"{c_cos:+.4f} cos({k}*theta)")
            if abs(c_sin) > coeff_threshold:
                terms.append(f"{c_sin:+.4f} sin({k}*theta)")

        if not terms:
            a_str = "0"
        else:
            a_str = " ".join(terms)
            if a_str.startswith("+"):
                a_str = a_str[1:]

        lines.append(f"  A_{j}(theta) = {a_str}")
        lines.append(f"  Fourier fit residual: {residual:.2e}")

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
    lines.append("  For a given theta, the QSP phases are:")
    lines.append(
        f"    phi_j = mean_phi_j"
        f" + sum_{{k=0}}^{{{n_dof-1}}} A_k(theta) * b_k[j]")
    lines.append("  Then the pulse schedule is built via phi_to_pulse_df(phi).")
    lines.append("=" * 70)

    if fidelity_results:
        lines.append("\n" + "=" * 70)
        lines.append("FIDELITY COMPARISON: NN (original) vs PCA (reconstructed)")
        lines.append("=" * 70)
        for theta in sorted(fidelity_results.keys()):
            fid = fidelity_results[theta]
            theta_label = f"{theta/math.pi:.2f}*pi"
            lines.append(
                f"  theta = {theta_label:>10s}  |  "
                f"NN = {fid['fidelity_nn']:.6f}  |  "
                f"PCA = {fid['fidelity_pca']:.6f}")
        lines.append("=" * 70)

    return "\n".join(lines)
