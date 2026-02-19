import hashlib
import json
import math
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from single_pulse_optimization_QSP.qsp_fit_relaxed import (
    TrainConfig, train, plot_matrix_element_vs_delta,
)
from single_pulse_optimization_QSP.qsp_fit import build_U
from single_pulse_optimization_QSP.qsp_fit_relaxed import build_U_with_detuning

from util import get_ore_ple_error_distribution, plot_pulse_param

try:
    from util import animate_multi_error_bloch
    _HAS_QUTIP = True
except Exception:
    _HAS_QUTIP = False


# ──────────────────────────────────────────────────────────────────────────────
#  Input parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_float_list(raw: str) -> Tuple[List[float], str]:
    cleaned = raw.replace("\n", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return [], "Enter a comma-separated list of numbers."
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            return [], f"Could not parse '{p}' as a number."
    return values, ""


# ──────────────────────────────────────────────────────────────────────────────
#  Phase cache helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_cache_key(
    delta_val_scaled: List[float],
    alpha_val_scaled: List[float],
    K: int,
    Delta_0_MHz: float,
    Omega_max_MHz: float,
    signal_window_MHz: float,
    build_with_detuning: bool,
    end_with_W: bool,
) -> str:
    """SHA-256 (first 16 hex chars) over the canonical JSON of all cache-relevant params."""
    payload = json.dumps(
        {
            "delta_vals": [round(v, 8) for v in delta_val_scaled],
            "alpha_vals": [round(v, 8) for v in alpha_val_scaled],
            "K": int(K),
            "Delta_0_MHz": round(Delta_0_MHz, 6),
            "Omega_max_MHz": round(Omega_max_MHz, 6),
            "signal_window_MHz": round(signal_window_MHz, 6),
            "build_with_detuning": bool(build_with_detuning),
            "end_with_W": bool(end_with_W),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_csv_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"phi_{key}.csv")


def _bloch_cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"bloch_{key}.mp4")


def _load_phase_cache(
    cache_dir: str, key: str
) -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """Return (phi_tensor, final_loss) if a cached CSV exists, else (None, None)."""
    path = _cache_csv_path(cache_dir, key)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    phi = torch.tensor(df["phi"].values, dtype=torch.float64)
    final_loss = float(df["final_loss"].iloc[0]) if "final_loss" in df.columns else float("nan")
    return phi, final_loss


def _save_phase_cache(
    cache_dir: str, key: str, phi: torch.Tensor, final_loss: float
) -> str:
    """Save phi values and final loss to a CSV; return the file path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_csv_path(cache_dir, key)
    data: dict = {"index": list(range(len(phi))), "phi": phi.numpy().tolist()}
    # Store final_loss only in the first row; remaining rows get NaN.
    loss_col = [float(final_loss)] + [float("nan")] * (len(phi) - 1)
    data["final_loss"] = loss_col
    pd.DataFrame(data).to_csv(path, index=False)
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  QSP visualization helpers
# ──────────────────────────────────────────────────────────────────────────────

def _spinor_to_bloch(psi: torch.Tensor) -> np.ndarray:
    a, b = psi[0], psi[1]
    return np.array([
        (2 * torch.real(torch.conj(a) * b)).item(),
        (2 * torch.imag(torch.conj(a) * b)).item(),
        (torch.abs(a) ** 2 - torch.abs(b) ** 2).item(),
    ])


def _qsp_unitary_batch(phi: torch.Tensor, delta_batch: torch.Tensor,
                        cfg: TrainConfig, omega_scale: float = 1.0) -> torch.Tensor:
    """Return (B,2,2) unitary for a batch of detuning values."""
    phi_cpu = phi.cpu()
    d_cpu = delta_batch.cpu()
    if cfg.build_with_detuning:
        U =  build_U_with_detuning(
            phi_cpu, d_cpu, cfg.Delta_0,
            cfg.Omega_max * omega_scale, end_with_W=cfg.end_with_W,
        )
    else:
        theta = (math.pi / 4) * (1 + d_cpu / cfg.Delta_0)
        U = build_U(phi_cpu, theta, end_with_W=cfg.end_with_W)
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    return H @ U.to(cfg.device) @ H

def _rx_target(alpha_i: float) -> torch.Tensor:
    """Build 2×2 Rx(alpha_i) = exp(-i * X * alpha / 2)."""
    a = torch.tensor(float(alpha_i), dtype=torch.float64)
    U = torch.zeros((2, 2), dtype=torch.complex128)
    U[0, 0] = torch.cos(0.5 * a)
    U[0, 1] = -1j * torch.sin(0.5 * a)
    U[1, 0] = -1j * torch.sin(0.5 * a)
    U[1, 1] = torch.cos(0.5 * a)
    return U


def _gate_fidelity_batch(phi: torch.Tensor, delta_batch: torch.Tensor,
                          alpha_i: float, cfg: TrainConfig,
                          omega_scale: float = 1.0) -> torch.Tensor:
    """Gate fidelity F = (|tr(U†_target U)|² + 2) / 6 for each delta in batch."""
    U = _qsp_unitary_batch(phi, delta_batch, cfg, omega_scale)
    Udagger = _rx_target(alpha_i).conj().T        # (2,2)
    traces = torch.einsum('ij,bji->b', Udagger, U)  # trace(U†_tgt @ U[b])
    return (traces.abs() ** 2 + 2.0) / 6.0


# 1. plot_pulse_param  ──────────────────────────────────────────────────────────
#    Directly callable from util.py with the existing pulse_df.
#    Called as:  plot_pulse_param(out_dir, title, pulse_df, Omega_max_scaled)


# 2. fidelity_contour_plot  ────────────────────────────────────────────────────

def qsp_fidelity_contour_plot(phi: torch.Tensor, cfg: TrainConfig,
                               delta_i: float, alpha_i: float,
                               out_path: str,
                               delta_vals: torch.Tensor,
                               alpha_vals: torch.Tensor,
                               N_delta: int = 100, N_eps: int = 40) -> None:
    """
    Fidelity contour over (detuning δ, calibration error ε) for one QSP target.
        delta range : [delta_i − sigma, delta_i + sigma]  (signal window)
        epsilon range: [-0.1, 0.1]  (±10% Omega_max calibration error)
    Title includes average gate fidelity across all peaks at ε = 0.
    """
    sigma = cfg.singal_window
    d_lo = max(-cfg.Delta_0, delta_i - sigma)
    d_hi = min(cfg.Delta_0, delta_i + sigma)
    delta_range = torch.linspace(d_lo, d_hi, N_delta, dtype=torch.float64)
    eps_vals = np.linspace(-0.1, 0.1, N_eps)

    F_grid = np.zeros((N_delta, N_eps))
    for j, eps in enumerate(eps_vals):
        F_grid[:, j] = _gate_fidelity_batch(
            phi, delta_range, alpha_i, cfg, omega_scale=1.0 + eps,
        ).numpy()

    # Average fidelity across all peaks at ε = 0
    # avg_F_per_peak = []
    # for dv, av in zip(delta_vals.tolist(), alpha_vals.tolist()):
    #     d_lo_p = max(-cfg.Delta_0, dv - sigma)
    #     d_hi_p = min(cfg.Delta_0, dv + sigma)
    #     d_samp = torch.linspace(d_lo_p, d_hi_p, 50, dtype=torch.float64)
    #     avg_F_per_peak.append(_gate_fidelity_batch(phi, d_samp, av, cfg).mean().item())
    avg_F = float(np.mean(F_grid))

    delta_MHz = delta_range.numpy() / (2 * math.pi)
    delta_i_MHz = delta_i / (2 * math.pi)
    sigma_MHz = sigma / (2 * math.pi)
    eps_pct = eps_vals * 100  # display as percent

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(delta_MHz, eps_pct, F_grid.T,
                     levels=[0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0], cmap='viridis')
    ax.contour(delta_MHz, eps_pct, F_grid.T,
               levels=[0.95, 0.99, 0.999], colors='white', linewidths=1.0)
    fig.colorbar(cf, ax=ax, label='Gate Fidelity')
    ax.axvline(delta_i_MHz - sigma_MHz, color='gray', ls=':', lw=1.2,
               label=f'δᵢ ± σ = {delta_i_MHz:.1f} ± {sigma_MHz:.1f} MHz')
    ax.axvline(delta_i_MHz + sigma_MHz, color='gray', ls=':', lw=1.2)
    ax.axhline(0, color='white', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel('Detuning δ (MHz)')
    ax.set_ylabel('Calibration error ε (%)')
    ax.set_title(
        f'Fidelity Contour: Rx({alpha_i/math.pi:.3f}π) '
        f'at δ = {delta_i_MHz:.1f} MHz\n'
        f'Avg F (uniform dist): {avg_F:.4f}'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# 3. plot_fidelity_by_std  ─────────────────────────────────────────────────────

def qsp_fidelity_by_std_plot(phi: torch.Tensor, cfg: TrainConfig,
                              delta_i: float, alpha_i: float,
                              out_path: str,
                              M: int = 2000, N_pts: int = 40) -> None:
    """
    Fidelity (and infidelity) vs std(δ) for one QSP target.
    Analog of plot_fidelity_by_std from util.py, adapted for QSP.
    """
    sigma_vals_rad = np.linspace(0.01 * cfg.Omega_max, 2.0 * cfg.Omega_max, N_pts)
    F_means, F_errs = [], []
    for sigma in sigma_vals_rad:
        delta_s = torch.randn(M, dtype=torch.float64) * sigma + delta_i
        delta_s = delta_s.clamp(-cfg.Delta_0, cfg.Delta_0)
        F = _gate_fidelity_batch(phi, delta_s, alpha_i, cfg)
        F_means.append(F.mean().item())
        F_errs.append(F.std().item() / math.sqrt(M))

    sigma_MHz = sigma_vals_rad / (2 * math.pi)
    inF = [1.0 - f for f in F_means]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].errorbar(sigma_MHz, F_means, yerr=F_errs, fmt='o-', capsize=4)
    axes[0].set_xlabel('std(δ) (MHz)')
    axes[0].set_ylabel('Expected Fidelity')
    axes[0].set_title(
        f'Fidelity vs δ spread\n'
        f'Rx({alpha_i/math.pi:.3f}π) at δ={delta_i/(2*math.pi):.1f} MHz'
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True)

    # Log-log infidelity plot (clip zeros to avoid log(0))
    inF_plot = [max(f, 1e-6) for f in inF]
    axes[1].errorbar(sigma_MHz, inF_plot, yerr=F_errs, fmt='o-', capsize=4)
    axes[1].set_xlabel('std(δ) (MHz)')
    axes[1].set_ylabel('Expected Infidelity')
    axes[1].set_title(
        f'Infidelity vs δ spread\n'
        f'Rx({alpha_i/math.pi:.3f}π) at δ={delta_i/(2*math.pi):.1f} MHz'
    )
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid(True, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


# 4. get_ore_ple_error_distribution  ───────────────────────────────────────────
#    Imported directly from util.py (used inside qsp_bloch_animation).


# 5. animate_multi_error_bloch  ────────────────────────────────────────────────

def _qsp_simulate_bloch(phi: torch.Tensor, delta_val: float,
                         cfg: TrainConfig) -> Tuple[np.ndarray, list]:
    """
    Simulate the full QSP sequence step-by-step for a single detuning value.
    Returns:
        bloch_vecs : (2K+2, 3) array of Bloch vectors (initial + one per gate)
        pulse_info : list of (0, phi_or_0, tau) tuples compatible with
                     animate_multi_error_bloch (phase_only=True)
    """
    from single_pulse_optimization_QSP.qsp_fit import W as _W

    psi = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    bloch_vecs = [_spinor_to_bloch(psi)]
    pulse_info = []

    theta_val = (math.pi / 4) * (1.0 + delta_val / cfg.Delta_0)
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2.0)

    W_mat = _W(torch.tensor([theta_val], dtype=torch.float64))[0].to(torch.complex128)

    W_mat = H @ W_mat @ H

    tau_W = math.pi / (4.0 * cfg.Delta_0)

    Omega = cfg.Omega_max
    norm = math.sqrt(Omega ** 2 + delta_val ** 2)


    # Iterate phi in reverse so the simulation applies phi[K] first and phi[0] last,
    # consistent with build_U with swapped axis: U = Rx(phi[0]) H W H … H W H Rx(phi[K]) applies Rz(phi[K])
    # first when acting on a ket.
    for j, pv in enumerate(reversed(phi.tolist())):
        # Rz step with detuning: H = ½(Ω σ_z + δ σ_x), t = pv/Ω
        lamb = pv * norm / (2.0 * Omega)
        c, s = math.cos(lamb), math.sin(lamb)
        off = complex(0.0, -s * delta_val / norm)

        U_rx = torch.zeros((2, 2), dtype=torch.complex128)
        U_rx[0, 0] = c
        U_rx[0, 1] = -1j * s * Omega / norm
        U_rx[1, 0] = -1j * s * Omega / norm
        U_rx[1, 1] = c
        # U_rz = torch.zeros((2, 2), dtype=torch.complex128)
        # U_rz[0, 0] = complex(c, -s * Omega / norm)
        # U_rz[1, 1] = complex(c, +s * Omega / norm)
        # U_rz[0, 1] = off
        # U_rz[1, 0] = off
        psi = U_rx @ psi
        bloch_vecs.append(_spinor_to_bloch(psi))
        # Pass rotation angle |pv| (not physical time); animate_multi_error_bloch
        # converts angle → time via Omega.
        pulse_info.append((0, pv, abs(pv)))

        if j < len(phi) - 1:
            psi = W_mat @ psi
            bloch_vecs.append(_spinor_to_bloch(psi))
            # W rotation-angle equivalent: Omega * tau_W
            pulse_info.append((0, 0.0, tau_W * Omega))

    return np.array(bloch_vecs), pulse_info


def qsp_bloch_animation(phi: torch.Tensor, cfg: TrainConfig,
                         delta_vals: torch.Tensor, alpha_vals: torch.Tensor,
                         out_path: str):
    """
    Bloch-sphere animation for 3×N qubits, one triplet per target peak:
        δᵢ − σ,  δᵢ,  δᵢ + σ
    where σ = cfg.singal_window.  Uses animate_multi_error_bloch from util.py.
    Returns (out_path, error_message).
    """
    if not _HAS_QUTIP:
        return None, (
            "qutip is not installed – Bloch animation unavailable. "
            "Run `pip install qutip` to enable this feature."
        )

    sigma = cfg.singal_window  # signal window in rad/s
    dv_cpu = delta_vals.cpu()
    av_cpu = alpha_vals.cpu()

    # Build the 3*N detuning list: [δ₀−σ, δ₀, δ₀+σ, δ₁−σ, δ₁, δ₁+σ, …]
    sim_deltas: List[float] = []
    for dv in dv_cpu.tolist():
        sim_deltas.extend([dv - sigma, dv, dv + sigma])

    # Clamp to the valid detuning range
    sim_deltas = [
        max(-cfg.Delta_0, min(cfg.Delta_0, d)) for d in sim_deltas
    ]

    n_qubits = len(sim_deltas)
    # Labels in MHz for the animation legend
    delta_MHz_labels = [d / (2 * math.pi) for d in sim_deltas]
    epsilon_list = [0.0] * n_qubits

    bloch_list, pinfo_list, fid_list = [], [], []

    for dv in sim_deltas:
        bvecs, pinfo = _qsp_simulate_bloch(phi, dv, cfg)
        bloch_list.append(bvecs)
        pinfo_list.append(pinfo)

        # Fidelity vs the nearest target Rx(αᵢ)
        ni = (dv_cpu - dv).abs().argmin().item()
        alpha_i = av_cpu[ni].item()
        F = _gate_fidelity_batch(phi, torch.tensor([dv], dtype=torch.float64),
                                  alpha_i, cfg)
        fid_list.append(F[0].item())

    # Build per-qubit legend labels:
    # "delta = <value> MHz, F = <fidelity>, target = Rx(<alpha/pi>π)"
    sigma_MHz = sigma / (2 * math.pi)
    label_list: List[str] = []
    for idx, d_MHz in enumerate(delta_MHz_labels):
        g = idx // 3          # which target peak
        av = av_cpu[g].item()
        target_str = f"Rx({av / math.pi:.3f}\u03c0)"
        label_list.append(
            f"delta = {d_MHz:.1f} MHz, F = {fid_list[idx]:.4f}, target = {target_str}"
        )

    N = len(dv_cpu)
    title = (
        f"QSP Ensemble Bloch Evolution — {N} peaks × 3 qubits "
        f"(δ, δ±σ={sigma_MHz:.1f} MHz)"
    )
    animate_multi_error_bloch(
        bloch_list, pinfo_list, fid_list,
        delta_MHz_labels, epsilon_list,
        name=title,
        save_path=out_path,
        phase_only=True,
        Omega=cfg.Omega_max / (2 * math.pi),
        label_list=label_list,
    )
    return out_path, None


# ──────────────────────────────────────────────────────────────────────────────
#  Page layout
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Controlled Unitary Demo", layout="wide")
st.title("Controlled Unitary Demo via QSP with Relaxed Constraints")

description = r"""
This demo allows you to generate control sequence for implementing target X-rotations
on a qubit using detuning as the control parameter. For a given list of delta_vals
and alpha_vals, we train a QSP sequence of phases to realize $R_x(\alpha_i)$ when detuning is near
$\delta_i$. Specifically, the target is to achieve

$R_x(\alpha_i)$ when $\delta = \delta_i \pm \sigma$.

for some window width $\sigma$ specified by `signal_window` = $\sigma$.

 Key parameters are

`N`: Number of target gates (or number of peaks in the detuning distribution)

`K`: Number of control gates. QSP sequence will generate polynomial $P(x)$ of degree at most $K$.

`Delta_0`: Maximum detuning (2$\pi$ MHz). Detuning distributions should be within $[-\Delta_0, \Delta_0]$.

`delta_vals`: List of detuning values (2$\pi$ MHz) at which target gates are defined.

`alpha_vals`: List of target rotation angles (in units of pi) corresponding to each delta

`signal_window`: Width of the detuning signal window (2$\pi$ MHz).

`build_with_detuning`: Whether to include detuning in the signal operator of QSP. When set to False, we are assuming infinitely fast control.
"""
st.markdown(description)

st.subheader("Disclaimer and Setup Instructions")

disclaimer = """
With `build_with_detuning` enabled, a reasonable `K` should be around 70.
However, the streamlit server will take a while to run (~30 min), and we recommend to run this demo locally (~10 min).
To do so, please follow the instruction below:

1. Clone the repository: `git clone https://github.com/shiminki/unitary_control_by_delta.git`
2. Enter the directory: `cd unitary_control_by_delta`
3. Install the necessary requirements: `pip install -r requirements.txt`
4. Run the streamlit app: `streamlit run streamlit_app.py`
"""
st.markdown(disclaimer)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Arguments")
    K = st.number_input("K (max phase index)", min_value=1, value=70, step=1)
    N = st.number_input("N (num_peaks = number of gates)", min_value=1, value=4, step=1)
    Delta_0_scaled = st.number_input("Delta_0 (MHz)", min_value=0.0, value=200.0, step=1.0)
    signal_window_scaled = st.number_input("signal_window (MHz)", min_value=0.0, value=10.0, step=0.1)
    Omega_max_scaled = st.number_input("Omega_max (MHz)", min_value=0.0, value=80.0, step=1.0)
    build_with_detuning = st.checkbox("build_with_detuning", value=True)


with col2:
    st.subheader("Other Arguments")
    steps = st.number_input("training steps", min_value=1, value=2000, step=100)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = st.number_input("lr", min_value=0.0, value=5e-2, step=1e-3, format="%.6f")
    device = st.selectbox("device", options=["cpu", "cuda"], index=0 if default_device == "cpu" else 1)
    end_with_W = st.checkbox("end_with_W", value=False)
    out_dir = st.text_input("out_dir", value="plots_relaxed")
    cache_dir = st.text_input(
        "cache_dir (leave blank to disable caching)",
        value="phase_cache",
        help="Directory where trained phases are cached as CSV files. "
             "Keyed by (delta_vals, alpha_vals, K, Omega_max, Delta_0, signal_window, "
             "build_with_detuning, end_with_W). Leave blank to always retrain.",
    )


st.subheader("Target Values")
alpha_default = "0.3333, 1, 0.5, 1.25"
delta_default = "-100, -32, 32, 100"
alpha_list_scaled = st.text_area("alpha_vals (radians unit in pi, length N)", value=alpha_default, height=80)
delta_list_scaled = st.text_area("delta_vals (MHz, length N)", value=delta_default, height=80)

run_btn = st.button("Run Training")

if "results" not in st.session_state:
    st.session_state["results"] = None
if "viz" not in st.session_state:
    st.session_state["viz"] = {}

# ──────────────────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────────────────

if run_btn:
    alpha_val_scaled, alpha_err = parse_float_list(alpha_list_scaled)
    delta_val_scaled, delta_err = parse_float_list(delta_list_scaled)

    errors = []
    if alpha_err:
        errors.append(f"alpha_vals: {alpha_err}")
    if delta_err:
        errors.append(f"delta_vals: {delta_err}")
    if not errors:
        if len(alpha_val_scaled) != N:
            errors.append(f"alpha_vals length is {len(alpha_val_scaled)}, expected N={N}.")
        if len(delta_val_scaled) != N:
            errors.append(f"delta_vals length is {len(delta_val_scaled)}, expected N={N}.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)

        if device == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA not available; falling back to CPU.")
            device = "cpu"

        cfg = TrainConfig(
            Omega_max=2 * math.pi * Omega_max_scaled,
            Delta_0=2 * math.pi * Delta_0_scaled,
            singal_window=2 * math.pi * signal_window_scaled,
            K=int(K),
            steps=int(steps),
            lr=float(lr),
            device=device,
            end_with_W=end_with_W,
            out_dir=out_dir,
            build_with_detuning=build_with_detuning,
        )

        delta_vals = torch.tensor(delta_val_scaled, device=device) * (2 * math.pi)
        alpha_vals = torch.tensor(alpha_val_scaled, device=device) * math.pi

        if (delta_vals.abs() > cfg.Delta_0).any():
            st.warning("Some delta_vals exceed |Delta_0| after unit conversion.")

        # ── Cache check ───────────────────────────────────────────────────────
        _use_cache = bool(cache_dir.strip())
        _cache_key = _make_cache_key(
            delta_val_scaled, alpha_val_scaled,
            int(K), float(Delta_0_scaled), float(Omega_max_scaled),
            float(signal_window_scaled), build_with_detuning, end_with_W,
        )
        _cached_phi, _cached_loss = (
            _load_phase_cache(cache_dir.strip(), _cache_key)
            if _use_cache else (None, None)
        )

        if _cached_phi is not None:
            phi_final = _cached_phi.to(device)
            final_loss = _cached_loss
            _cache_csv = _cache_csv_path(cache_dir.strip(), _cache_key)
            st.success(
                f"Cache hit — skipping training. "
                f"Loaded phases from `{_cache_csv}` (loss: {final_loss:.3e})."
            )
        else:
            with st.spinner("Training..."):
                progress_bar = st.progress(0)
                progress_text = st.empty()

                def progress_cb(step: int, total: int, loss: float, eta: float) -> None:
                    progress_bar.progress(step / total)
                    eta_min = int(eta // 60)
                    eta_sec = int(eta % 60)
                    progress_text.write(
                        f"Step {step}/{total} — loss {loss:.3e} — ETA {eta_min:02d}:{eta_sec:02d}"
                    )

                phi_final, final_loss = train(
                    cfg,
                    delta_vals,
                    alpha_vals,
                    sample_size=2048,
                    progress_cb=progress_cb,
                    verbose=True,
                )

            if _use_cache:
                _saved_path = _save_phase_cache(
                    cache_dir.strip(), _cache_key, phi_final.cpu(), final_loss
                )
                st.info(f"Phases saved to cache: `{_saved_path}` (key: `{_cache_key}`).")

        # Build pulse schedule DataFrame
        tau_us = math.pi / (4.0 * cfg.Delta_0)
        t_rows, hx_rows, hz_rows = [], [], []
        for i, phi in enumerate(phi_final.tolist()):
            t_rows.append(np.abs(phi) / cfg.Omega_max)
            hx_rows.append(Omega_max_scaled * np.sign(phi))
            hz_rows.append(0.0)
            if i != len(phi_final) - 1:
                t_rows.append(tau_us)
                hx_rows.append(0.0)
                hz_rows.append(Delta_0_scaled)

        pulse_df = pd.DataFrame({
            "t (us)": t_rows,
            "Omega_x (2pi MHz)": hx_rows,
            "Omega_y (2pi MHz)": [0.0] * len(hx_rows),
            "Omega_z (2pi MHz)": hz_rows,
        })

        plot_path = os.path.join(out_dir, f"u00_final_K={int(K)}.png")
        plot_matrix_element_vs_delta(phi_final, cfg, delta_vals, alpha_vals, out_path=plot_path)

        st.session_state["results"] = {
            "final_loss": final_loss,
            "pulse_df": pulse_df,
            "plot_path": plot_path,
            "phi_final": phi_final,          # stored for visualizations
            "delta_vals": delta_vals.cpu(),  # stored for visualizations
            "alpha_vals": alpha_vals.cpu(),  # stored for visualizations
            "cfg": cfg,
            "cache_key": _cache_key,         # for animation / viz caching
            "cache_dir": cache_dir.strip(),  # for animation / viz caching
            "args": {
                "K": int(K),
                "steps": int(steps),
                "num_peaks": int(N),
                "Delta_0": float(Delta_0_scaled),
                "signal_window": float(signal_window_scaled),
                "Omega_max": float(Omega_max_scaled),
                "lr": float(lr),
                "device": device,
                "end_with_W": end_with_W,
                "out_dir": out_dir,
                "build_with_detuning": build_with_detuning,
            },
        }
        # Clear cached visualizations when new training runs
        st.session_state["viz"] = {}

# ──────────────────────────────────────────────────────────────────────────────
#  Results & Visualizations
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state["results"] is not None:
    results = st.session_state["results"]
    viz = st.session_state["viz"]

    st.success(f"Training complete. Final loss: {results['final_loss']:.6e}")

    # Recover stored tensors / config
    phi_final = results["phi_final"]
    delta_vals_cpu = results["delta_vals"]
    alpha_vals_cpu = results["alpha_vals"]
    cfg_stored = results["cfg"]
    args = results["args"]
    out_dir_stored = args["out_dir"]
    os.makedirs(out_dir_stored, exist_ok=True)

    # Build human-readable target labels for selectboxes
    target_labels = [
        f"Target {i}: δ={delta_vals_cpu[i].item()/(2*math.pi):.1f} MHz, "
        f"Rx({alpha_vals_cpu[i].item()/math.pi:.3f}π)"
        for i in range(len(delta_vals_cpu))
    ]

    # ── Tabs ──────────────────────────────────────────────────────────────────
    (tab_pulse, tab_matrix, tab_contour,
     tab_fid_std, tab_bloch) = st.tabs([
        "1 · Pulse Schedule",
        "2 · Matrix Element vs δ",
        "3 · Fidelity Contour",
        "4 · Fidelity vs Std",
        "5 · Bloch Animation",
    ])

    # ── Tab 1: Pulse Schedule ─────────────────────────────────────────────────
    with tab_pulse:
        st.subheader("Pulse Schedule")
        st.dataframe(results["pulse_df"], use_container_width=True)
        st.download_button(
            label="Download pulse schedule CSV",
            data=results["pulse_df"].to_csv(index=False).encode("utf-8"),
            file_name="pulse_schedule.csv",
            mime="text/csv",
            key="download_pulse_csv",
        )

        pulse_param_path = os.path.join(out_dir_stored, "pulse_param_plot.png")
        if "pulse_param_path" not in viz or not os.path.exists(pulse_param_path):
            with st.spinner("Generating pulse schedule plot…"):
                plot_pulse_param(
                    out_dir_stored,
                    f"QSP_K{args['K']}",
                    results["pulse_df"],
                    args["Omega_max"],
                )
                generated = os.path.join(out_dir_stored, f"QSP_K{args['K']}.png")
                if os.path.exists(generated):
                    os.rename(generated, pulse_param_path)
            viz["pulse_param_path"] = pulse_param_path
            st.session_state["viz"] = viz

        if os.path.exists(pulse_param_path):
            st.image(pulse_param_path, caption="Pulse schedule (Ωₓ, Ωᵧ, Ω_z vs time)",
                     use_column_width=True)

    # ── Tab 2: Matrix Element vs δ ────────────────────────────────────────────
    with tab_matrix:
        st.subheader("Matrix Element vs Detuning")
        if os.path.exists(results["plot_path"]):
            st.image(results["plot_path"],
                     caption="Re(u₀₀) and Im(u₀₁) vs δ with target windows",
                     use_column_width=True)
        else:
            st.info(f"Plot not found at `{results['plot_path']}`")
        st.write("Args used:", args)

    # ── Tab 3: Fidelity Contour ───────────────────────────────────────────────
    with tab_contour:
        st.subheader("Fidelity Contour Plot")
        st.markdown(
            "Gate fidelity of Rx(αᵢ) over detuning δ ∈ [δᵢ − σ, δᵢ + σ] "
            "and calibration error ε ∈ [−10%, +10%] (Ω_max scaled by 1 + ε). "
            "Title shows the average fidelity across all detuning peaks at ε = 0."
        )
        sel_c = st.selectbox("Select target", target_labels, key="sel_contour")
        idx_c = target_labels.index(sel_c)

        if st.button("Generate Fidelity Contour", key="btn_contour"):
            contour_path = os.path.join(out_dir_stored, f"fidelity_contour_target{idx_c}.png")
            with st.spinner("Computing fidelity contour…"):
                qsp_fidelity_contour_plot(
                    phi_final, cfg_stored,
                    delta_vals_cpu[idx_c].item(),
                    alpha_vals_cpu[idx_c].item(),
                    contour_path,
                    delta_vals_cpu,
                    alpha_vals_cpu,
                )
            viz[f"contour_{idx_c}"] = contour_path
            st.session_state["viz"] = viz

        cache_key = f"contour_{idx_c}"
        if cache_key in viz and os.path.exists(viz[cache_key]):
            st.image(viz[cache_key], caption="Fidelity contour", use_column_width=True)

    # ── Tab 4: Fidelity vs Std ────────────────────────────────────────────────
    with tab_fid_std:
        st.subheader("Fidelity vs δ Spread")
        st.markdown(
            "Expected gate fidelity as a function of the std of the detuning distribution "
            "centred at the target δᵢ. Analog of `plot_fidelity_by_std` from `util.py`."
        )
        sel_f = st.selectbox("Select target", target_labels, key="sel_fid_std")
        idx_f = target_labels.index(sel_f)
        M_fid = st.number_input("Monte-Carlo samples M", min_value=100, value=2000, step=100,
                                 key="M_fid")

        if st.button("Generate Fidelity vs Std", key="btn_fid_std"):
            fid_std_path = os.path.join(out_dir_stored, f"fidelity_by_std_target{idx_f}.png")
            with st.spinner("Computing fidelity vs std…"):
                qsp_fidelity_by_std_plot(
                    phi_final, cfg_stored,
                    delta_vals_cpu[idx_f].item(),
                    alpha_vals_cpu[idx_f].item(),
                    fid_std_path,
                    M=int(M_fid),
                )
            viz[f"fid_std_{idx_f}"] = fid_std_path
            st.session_state["viz"] = viz

        cache_key_f = f"fid_std_{idx_f}"
        if cache_key_f in viz and os.path.exists(viz[cache_key_f]):
            st.image(viz[cache_key_f], caption="Fidelity vs std(δ)", use_column_width=True)

    # ── Tab 5: Bloch Animation ────────────────────────────────────────────────
    with tab_bloch:
        st.subheader("Ensemble Bloch Sphere Animation")
        N_peaks = len(delta_vals_cpu)
        sigma_MHz = cfg_stored.singal_window / (2 * math.pi)
        st.markdown(
            f"Qubit-state trajectories for **{3 * N_peaks} qubits** "
            f"({N_peaks} peaks × 3): one triplet per target peak at "
            f"δᵢ − σ, δᵢ, δᵢ + σ with σ = {sigma_MHz:.1f} MHz. "
            "Calls `animate_multi_error_bloch` from `util.py`."
        )
        if not _HAS_QUTIP:
            st.warning(
                "qutip is not installed. Install it with `pip install qutip` to enable "
                "the Bloch sphere animation."
            )
        else:
            # Resolve the cache-aware animation path for this result set
            _res_cache_key = results.get("cache_key", "")
            _res_cache_dir = results.get("cache_dir", "")
            _use_anim_cache = bool(_res_cache_key and _res_cache_dir)
            _cached_anim = (
                _bloch_cache_path(_res_cache_dir, _res_cache_key)
                if _use_anim_cache else ""
            )

            # Auto-populate viz from disk cache on first render (avoids re-render)
            if _use_anim_cache and os.path.exists(_cached_anim) and "bloch_path" not in viz:
                viz["bloch_path"] = _cached_anim
                st.session_state["viz"] = viz

            if st.button("Generate Bloch Animation", key="btn_bloch"):
                if _use_anim_cache and os.path.exists(_cached_anim):
                    # Cache hit — no rendering needed
                    viz["bloch_path"] = _cached_anim
                    st.session_state["viz"] = viz
                    st.success(
                        f"Cache hit — loaded animation from `{_cached_anim}`."
                    )
                else:
                    # Save directly to cache path when caching is enabled
                    anim_path = (
                        _cached_anim if _use_anim_cache
                        else os.path.join(out_dir_stored, "bloch_animation.mp4")
                    )
                    with st.spinner(
                        f"Simulating {3 * N_peaks} qubit trajectories and rendering animation… "
                        "(this may take a minute)"
                    ):
                        out, err_msg = qsp_bloch_animation(
                            phi_final, cfg_stored,
                            delta_vals_cpu, alpha_vals_cpu,
                            anim_path,
                        )
                    if err_msg:
                        st.error(err_msg)
                    elif out and os.path.exists(out):
                        viz["bloch_path"] = out
                        st.session_state["viz"] = viz
                        if _use_anim_cache:
                            st.info(f"Animation cached at `{out}` (key: `{_res_cache_key}`).")

            if "bloch_path" in viz and os.path.exists(viz["bloch_path"]):
                st.video(viz["bloch_path"])
                with open(viz["bloch_path"], "rb") as vf:
                    st.download_button(
                        label="Download animation (mp4)",
                        data=vf.read(),
                        file_name="bloch_animation.mp4",
                        mime="video/mp4",
                        key="download_bloch",
                    )
