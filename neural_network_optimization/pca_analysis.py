"""Per-peak PCA of the PulseNet phi trajectory.

For peak ``p``, the alpha input is ``[0, ..., alpha at index p, ..., 0]`` with
``alpha`` swept over ``[0, 4*pi]`` (full SU(2) period).  phi(alpha) is
extracted from the trained network, PCA'd, and each amplitude is fit with
Fourier and polynomial series.
"""

import argparse
import math
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .constants import DEFAULT_K, N_PEAKS, OMEGA_MHZ
from .inference import load_model, load_single_peak_model
from .model import PulseNet, SinglePeakNet


# ─────────────────────────────────────────────────────────────────────────────
#  Amplitude fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fit_fourier(alpha_np, amplitudes, n_dof, n_fourier_max: int = 8):
    """Fit each amplitude A_j(alpha) with a Fourier series in alpha/2.

    Half-angle basis captures the 4*pi period of SU(2).
    """
    N = len(alpha_np)
    fits = []
    for j in range(n_dof):
        amp_j = amplitudes[:, j]
        n_fourier = min(n_fourier_max, max(1, N // 4))
        design = [np.ones(N)]
        for k in range(1, n_fourier + 1):
            design.append(np.cos(k * alpha_np / 2))
            design.append(np.sin(k * alpha_np / 2))
        design = np.column_stack(design)
        coeffs, _, _, _ = np.linalg.lstsq(design, amp_j, rcond=None)
        residual = float(np.mean((amp_j - design @ coeffs) ** 2))
        fits.append({"coeffs": coeffs, "n_fourier": n_fourier, "residual": residual})
    return fits


def _fit_polynomial(alpha_np, amplitudes, n_dof, max_deg: int = 10):
    """Fit each amplitude with the polynomial of degree 1..max_deg that minimises AIC."""
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
        rss = float(np.sum((amp_k - np.polyval(best_coeffs, alpha_np)) ** 2))
        ss_tot = float(np.sum((amp_k - amp_k.mean()) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 1.0
        fits.append((best_deg, best_coeffs, r2))
    return fits


# ─────────────────────────────────────────────────────────────────────────────
#  Reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_phi_analytical(
    pca_result: Dict, alpha: float, n_basis: Optional[int] = None,
) -> np.ndarray:
    """phi(alpha) reconstructed from PCA + Fourier fits."""
    mean_phi = pca_result["mean_phi"]
    pc = pca_result["principal_components"]
    fits = pca_result["amplitude_fits"]
    n_dof = pca_result["n_effective_dof"]
    if n_basis is None:
        n_basis = n_dof
    n_basis = min(n_basis, n_dof)
    phi = mean_phi.copy()
    for j in range(n_basis):
        coeffs = fits[j]["coeffs"]
        n_fourier = fits[j]["n_fourier"]
        val = coeffs[0]
        for k in range(1, n_fourier + 1):
            val += coeffs[2 * k - 1] * math.cos(k * alpha / 2)
            val += coeffs[2 * k] * math.sin(k * alpha / 2)
        phi = phi + val * pc[:, j]
    return phi


def reconstruct_phi_polyfit(
    pca_result: Dict, alpha: float, n_basis: Optional[int] = None,
) -> np.ndarray:
    """phi(alpha) reconstructed from PCA + polynomial fits."""
    mean_phi = pca_result["mean_phi"]
    pc = pca_result["principal_components"]
    fits = pca_result["polyfit_fits"]
    n_dof = pca_result["n_effective_dof"]
    if n_basis is None:
        n_basis = n_dof
    n_basis = min(n_basis, n_dof)
    phi = mean_phi.copy()
    for j in range(n_basis):
        _deg, coeffs, _r2 = fits[j]
        val = float(np.polyval(coeffs, alpha))
        phi = phi + val * pc[:, j]
    return phi


# ─────────────────────────────────────────────────────────────────────────────
#  CSVs + analytical-form text
# ─────────────────────────────────────────────────────────────────────────────

def save_pca_csv(pca_result: Dict, peak_index: int, out_dir: str):
    """Emit basis_functions, fourier_coefficients, and polyfit_coefficients CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    pc = pca_result["principal_components"]
    n_dof = pca_result["n_effective_dof"]
    mean_phi = pca_result["mean_phi"]

    basis_rows = {"phase_index": np.arange(pc.shape[0]), "mean_phi": mean_phi}
    for j in range(n_dof):
        basis_rows[f"b_{j}"] = pc[:, j]
    pd.DataFrame(basis_rows).to_csv(
        os.path.join(out_dir, f"basis_functions_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )

    # Fourier
    f_rows = []
    for j in range(n_dof):
        coeffs = pca_result["amplitude_fits"][j]["coeffs"]
        n_f = pca_result["amplitude_fits"][j]["n_fourier"]
        row = {"component": j, "a_0": coeffs[0]}
        for k in range(1, n_f + 1):
            row[f"a_cos{k}"] = coeffs[2 * k - 1]
            row[f"a_sin{k}"] = coeffs[2 * k]
        row["residual"] = pca_result["amplitude_fits"][j]["residual"]
        f_rows.append(row)
    pd.DataFrame(f_rows).to_csv(
        os.path.join(out_dir, f"fourier_coefficients_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )

    # Polynomial
    p_rows = []
    max_deg = max(deg for deg, _, _ in pca_result["polyfit_fits"])
    for j, (deg, coeffs_desc, r2) in enumerate(pca_result["polyfit_fits"]):
        coeffs_asc = coeffs_desc[::-1]
        row = {"component": j, "degree": deg, "R2": round(r2, 8)}
        for i in range(max_deg + 1):
            row[f"c_{i}"] = float(coeffs_asc[i]) if i <= deg else 0.0
        p_rows.append(row)
    pd.DataFrame(p_rows).to_csv(
        os.path.join(out_dir, f"polyfit_coefficients_peak{peak_index}.csv"),
        index=False, float_format="%.8f",
    )


def format_analytical_form(
    pca_result: Dict,
    fit_type: str = "fourier_fit",
    coeff_threshold: float = 1e-3,
    fidelity_results: Optional[Dict] = None,
) -> str:
    """Produce a human-readable decomposition string for the plotting tab / txt output."""
    n_dof = pca_result["n_effective_dof"]
    pc = pca_result["principal_components"]
    mean_phi = pca_result["mean_phi"]
    sv = pca_result["singular_values"]
    ev = pca_result["explained_variance_ratio"]
    K = pc.shape[0] - 1

    fit_label = "Fourier" if fit_type == "fourier_fit" else "Polynomial"

    lines = ["=" * 70,
             f"ANALYTICAL DECOMPOSITION ({fit_label})",
             f"  phi(alpha) = phi_mean + sum_{{j=0}}^{{{n_dof-1}}} A_j(alpha) * b_j",
             f"  Effective components d = {n_dof}   (K = {K})",
             f"  Explained variance  = {sum(ev[:n_dof]) * 100:.2f}%",
             "=" * 70,
             f"\n||phi_mean|| = {np.linalg.norm(mean_phi):.4f}"]
    for j in range(n_dof):
        lines.append("\n" + "-" * 50)
        lines.append(f"j = {j}   sigma={sv[j]:.4f}   var={ev[j]*100:.2f}%")
        lines.append("-" * 50)
        if fit_type == "fourier_fit":
            fit = pca_result["amplitude_fits"][j]
            coeffs = fit["coeffs"]
            n_f = fit["n_fourier"]
            terms = []
            if abs(coeffs[0]) > coeff_threshold:
                terms.append(f"{coeffs[0]:+.4f}")
            for k in range(1, n_f + 1):
                cc, cs = coeffs[2 * k - 1], coeffs[2 * k]
                if abs(cc) > coeff_threshold:
                    terms.append(f"{cc:+.4f} cos({k}*alpha/2)")
                if abs(cs) > coeff_threshold:
                    terms.append(f"{cs:+.4f} sin({k}*alpha/2)")
            body = " ".join(terms) if terms else "0"
            lines.append(f"  A_{j}(alpha) = {body.lstrip('+')}")
            lines.append(f"  Fourier residual: {fit['residual']:.2e}")
        else:
            deg, coeffs_desc, r2 = pca_result["polyfit_fits"][j]
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
            body = " ".join(terms) if terms else "0"
            lines.append(f"  A_{j}(alpha) = {body.lstrip('+')}")
            lines.append(f"  Polynomial: deg={deg}, R^2={r2:.6f}")

    if fidelity_results:
        lines.append("\n" + "=" * 70)
        lines.append("FIDELITY: NN vs reconstruction")
        lines.append("=" * 70)
        for a in sorted(fidelity_results):
            fr = fidelity_results[a]
            lines.append(
                f"  alpha = {a/math.pi:.3f}π   NN={fr['fidelity_nn']:.6f}   "
                f"Fit={fr['fidelity_fit']:.6f}"
            )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Core PCA driver
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(
    model: PulseNet,
    peak_index: int,
    *,
    n_samples: int = 512,
    explained_var_threshold: float = 0.999,
    polyfit_max_deg: int = 10,
    out_dir: Optional[str] = None,
    save_figures: bool = True,
    alpha_max: float = 4 * math.pi,
) -> Dict:
    """Run per-peak PCA on ``model([0, ..., alpha, ..., 0])`` and (optionally) save artefacts.

    Returns a dict consumed by ``plotting.plot_pca_*`` and the reconstruction helpers.
    """
    assert 0 <= peak_index < model.N_peaks
    model.eval()
    dev = next(model.parameters()).device

    alpha_grid_t = torch.linspace(0.0, alpha_max, n_samples, dtype=torch.float64, device=dev)
    batch = torch.zeros(n_samples, model.N_peaks, dtype=torch.float64, device=dev)
    batch[:, peak_index] = alpha_grid_t

    with torch.no_grad():
        phi_matrix_t = model(batch)  # (n_samples, K+1)
    phi_matrix = phi_matrix_t.detach().cpu().numpy()
    alpha_np = alpha_grid_t.detach().cpu().numpy()

    mean_phi = phi_matrix.mean(axis=0)
    phi_centered = phi_matrix - mean_phi

    _, S, Vt = np.linalg.svd(phi_centered, full_matrices=False)
    var = S ** 2
    total = var.sum()
    explained_ratio = var / total if total > 0 else var
    cumulative = np.cumsum(explained_ratio)
    n_dof = int(np.searchsorted(cumulative, explained_var_threshold) + 1)
    n_dof = min(max(n_dof, 1), len(S))

    pc = Vt[:n_dof, :].T  # (K+1, n_dof)
    amplitudes = phi_centered @ pc  # (n_samples, n_dof)

    amplitude_fits = _fit_fourier(alpha_np, amplitudes, n_dof)
    polyfit_fits = _fit_polynomial(alpha_np, amplitudes, n_dof, max_deg=polyfit_max_deg)

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
        "peak_index": peak_index,
    }

    if save_figures and out_dir is not None:
        _save_pca_artifacts(model, result, peak_index, out_dir)

    return result


def _save_pca_artifacts(model: PulseNet, pca_result: Dict, peak_index: int, out_dir: str):
    """Write the four PNGs, three CSVs, and the analytical-form text file."""
    from .plotting import (
        plot_amplitude_components,
        plot_comparative_matrix_elements,
        plot_comparative_pulses,
        plot_pca_overview,
    )

    peak_dir = os.path.join(out_dir, f"peak{peak_index}")
    os.makedirs(peak_dir, exist_ok=True)

    plot_pca_overview(
        pca_result, peak_index,
        os.path.join(peak_dir, f"pca_overview_peak{peak_index}.png"),
    )
    plot_amplitude_components(
        pca_result, peak_index,
        os.path.join(peak_dir, f"amplitude_components_peak{peak_index}.png"),
    )
    plot_comparative_pulses(
        model, pca_result, peak_index,
        os.path.join(peak_dir, f"comparative_pulses_peak{peak_index}.png"),
    )
    fidelity_results = plot_comparative_matrix_elements(
        model, pca_result, peak_index,
        os.path.join(peak_dir, f"comparative_matrix_elements_peak{peak_index}.png"),
    )
    save_pca_csv(pca_result, peak_index, peak_dir)

    text = format_analytical_form(
        pca_result, fit_type="fourier_fit", fidelity_results=fidelity_results,
    )
    with open(os.path.join(peak_dir, f"analytical_form_peak{peak_index}.txt"), "w") as f:
        f.write(text)


def run_pca_all_peaks(
    model: PulseNet,
    peak_indices=None,
    out_dir: str = "outputs/pca",
    **kwargs,
) -> Dict[int, Dict]:
    """Run PCA for each peak and return dict {peak_index -> pca_result}."""
    if peak_indices is None:
        peak_indices = list(range(model.N_peaks))
    results = {}
    for p in peak_indices:
        results[p] = run_pca(model, p, out_dir=out_dir, **kwargs)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Omega", type=float, default=OMEGA_MHZ)
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--weight_dir", type=str, default="neural_network_optimization/weights")
    ap.add_argument("--out_dir", type=str, default="outputs/pca")
    ap.add_argument("--n_samples", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--single_peak", action="store_true",
                    help="Run PCA on a SinglePeakNet instead of the full joint PulseNet.")
    ap.add_argument("--peak_index", type=int, default=0,
                    help="Which peak the SinglePeakNet targets (default: 0).")
    ap.add_argument("--train_if_missing", action="store_true",
                    help="Auto-train the model if no checkpoint is found.")
    # Training hyper-params forwarded when --train_if_missing triggers training.
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.single_peak:
        try:
            model = load_single_peak_model(
                Omega=args.Omega, K=args.K, peak_index=args.peak_index,
                weight_dir=args.weight_dir, device=args.device,
            )
        except FileNotFoundError as exc:
            if not args.train_if_missing:
                raise
            print(f"[pca_analysis] Checkpoint not found — training now.\n  ({exc})")
            from .train import train_single_peak
            model = train_single_peak(
                Omega=args.Omega, K=args.K, peak_index=args.peak_index,
                epochs=args.epochs, lr=args.lr, seed=args.seed,
                weight_dir=args.weight_dir, device=args.device,
            )

        out_dir = os.path.join(
            args.out_dir,
            f"single_peak{args.peak_index}_Omega{args.Omega}_K{args.K}",
        )
        run_pca(
            model,
            peak_index=args.peak_index,
            out_dir=out_dir,
            n_samples=args.n_samples,
        )
    else:
        model = load_model(Omega=args.Omega, K=args.K, weight_dir=args.weight_dir, device=args.device)
        run_pca_all_peaks(
            model,
            peak_indices=list(range(model.N_peaks)),
            out_dir=os.path.join(args.out_dir, f"Omega{args.Omega}_K{args.K}"),
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
