"""Streamlit demo using the pretrained joint neural network for pulse generation.

Same structure as the original streamlit_app.py (5 tabs), but uses neural network
inference instead of gradient-based QSP training for each input.

Usage:
    streamlit run neural_network_optimization/streamlit_app.py
"""

import math
import os
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from neural_network_optimization.model import (
    JointPulseGeneratorNet,
    phi_to_pulse_df,
    get_weight_path,
    DELTA_CENTERS_MHZ,
    N_PEAKS,
    DEFAULT_K,
    DEFAULT_OMEGA_MHZ,
    DELTA_0_MHZ,
    ROBUSTNESS_WINDOW_MHZ,
)
from neural_network_optimization.inference import (
    load_model,
    generate_pulse,
    generate_phi,
    compute_fidelity,
    compute_runtime,
)
from neural_network_optimization.train import train as train_model

from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    build_qsp_unitary,
    delta_to_theta,
    signal_operator,
    get_wait_time,
)
from util import plot_pulse_param

try:
    from util import animate_multi_error_bloch
    _HAS_QUTIP = True
except Exception:
    _HAS_QUTIP = False


# ──────────────────────────────────────────────────────────────────────────────
#  Visualization helpers (adapted from streamlit_app.py for phi-based input)
# ──────────────────────────────────────────────────────────────────────────────

def _rx_target(alpha_i: float) -> torch.Tensor:
    """Build 2x2 Rx(alpha_i) = exp(-i * X * alpha / 2)."""
    a = torch.tensor(float(alpha_i), dtype=torch.float64)
    U = torch.zeros((2, 2), dtype=torch.complex128)
    U[0, 0] = torch.cos(0.5 * a)
    U[0, 1] = -1j * torch.sin(0.5 * a)
    U[1, 0] = -1j * torch.sin(0.5 * a)
    U[1, 1] = torch.cos(0.5 * a)
    return U


def _qsp_unitary_batch(phi, delta_batch, Delta_0_ang, Omega_ang):
    """Return (B,2,2) physical-basis unitary for a batch of detuning values."""
    U = build_qsp_unitary(phi.cpu(), delta_batch.cpu(), Delta_0_ang, Omega_ang)
    H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    return H @ U @ H


def _gate_fidelity_batch(phi, delta_batch, alpha_i, Delta_0_ang, Omega_ang):
    """Gate fidelity F = (|tr(U_target^dag U)|^2 + 2) / 6 for each delta."""
    U = _qsp_unitary_batch(phi, delta_batch, Delta_0_ang, Omega_ang)
    Udagger = _rx_target(alpha_i).conj().T
    traces = torch.einsum('ij,bji->b', Udagger, U)
    return (traces.abs() ** 2 + 2.0) / 6.0


def nn_matrix_element_plot(phi, Delta_0_ang, Omega_ang, delta_vals_ang,
                           alpha_vals_rad, sigma_ang, out_path):
    """Plot Re(u_00) and Im(u_01) vs detuning with target windows."""
    delta_range = torch.linspace(-Delta_0_ang, Delta_0_ang, 1024, dtype=torch.float64)
    U = _qsp_unitary_batch(phi, delta_range, Delta_0_ang, Omega_ang)
    u00 = U[:, 0, 0].numpy()
    u01 = U[:, 0, 1].numpy()
    delta_mhz = (delta_range / (2 * math.pi)).numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(delta_mhz, u00.real, label="Re(u₀₀)", linewidth=1.2)
    ax.plot(delta_mhz, u01.imag, label="Im(u₀₁)", linewidth=1.2)

    for i, (dv, av) in enumerate(zip(delta_vals_ang, alpha_vals_rad)):
        dv_mhz = dv / (2 * math.pi)
        s_mhz = sigma_ang / (2 * math.pi)
        ax.hlines(y=np.cos(av / 2), xmin=dv_mhz - s_mhz, xmax=dv_mhz + s_mhz,
                  colors="red", linestyles="dotted", linewidth=1.5,
                  label="target Re(u₀₀)" if i == 0 else None)
        ax.hlines(y=-np.sin(av / 2), xmin=dv_mhz - s_mhz, xmax=dv_mhz + s_mhz,
                  colors="green", linestyles="dotted", linewidth=1.5,
                  label="target Im(u₀₁)" if i == 0 else None)
        ax.axvspan(dv_mhz - s_mhz, dv_mhz + s_mhz, color="gray", alpha=0.1)

    ax.set_xlabel("Detuning δ (MHz)")
    ax.set_ylabel("Matrix Element")
    ax.set_ylim(-1.3, 1.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Matrix Elements vs Detuning (Physical Basis)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def nn_fidelity_contour_plot(phi, Delta_0_ang, Omega_ang, delta_i_ang, alpha_i,
                              sigma_ang, out_path, N_delta=100, N_eps=40):
    """Fidelity contour over (detuning, calibration error) for one peak."""
    d_lo = max(-Delta_0_ang, delta_i_ang - sigma_ang)
    d_hi = min(Delta_0_ang, delta_i_ang + sigma_ang)
    delta_range = torch.linspace(d_lo, d_hi, N_delta, dtype=torch.float64)
    eps_vals = np.linspace(-0.1, 0.1, N_eps)

    F_grid = np.zeros((N_delta, N_eps))
    for j, eps in enumerate(eps_vals):
        Omega_scaled = Omega_ang * (1.0 + eps)
        F_grid[:, j] = _gate_fidelity_batch(
            phi, delta_range, alpha_i, Delta_0_ang, Omega_scaled,
        ).numpy()

    avg_F = float(np.mean(F_grid))
    delta_MHz = (delta_range / (2 * math.pi)).numpy()
    delta_i_MHz = delta_i_ang / (2 * math.pi)
    sigma_MHz = sigma_ang / (2 * math.pi)
    eps_pct = eps_vals * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(delta_MHz, eps_pct, F_grid.T,
                     levels=[0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0],
                     cmap='viridis')
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


def nn_fidelity_by_std_plot(phi, Delta_0_ang, Omega_ang, delta_i_ang, alpha_i,
                             out_path, M=2000, N_pts=40):
    """Fidelity (and infidelity) vs std(δ) for one peak."""
    sigma_vals_rad = np.linspace(0.01 * Omega_ang, 2.0 * Omega_ang, N_pts)
    F_means, F_errs = [], []
    for sigma in sigma_vals_rad:
        delta_s = torch.randn(M, dtype=torch.float64) * sigma + delta_i_ang
        delta_s = delta_s.clamp(-Delta_0_ang, Delta_0_ang)
        F = _gate_fidelity_batch(phi, delta_s, alpha_i, Delta_0_ang, Omega_ang)
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
        f'Rx({alpha_i/math.pi:.3f}π) at δ={delta_i_ang/(2*math.pi):.1f} MHz'
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True)

    inF_plot = [max(f, 1e-6) for f in inF]
    axes[1].errorbar(sigma_MHz, inF_plot, yerr=F_errs, fmt='o-', capsize=4)
    axes[1].set_xlabel('std(δ) (MHz)')
    axes[1].set_ylabel('Expected Infidelity')
    axes[1].set_title(
        f'Infidelity vs δ spread\n'
        f'Rx({alpha_i/math.pi:.3f}π) at δ={delta_i_ang/(2*math.pi):.1f} MHz'
    )
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    axes[1].grid(True, which='both')

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def _spinor_to_bloch(psi: torch.Tensor) -> np.ndarray:
    a, b = psi[0], psi[1]
    return np.array([
        (2 * torch.real(torch.conj(a) * b)).item(),
        (2 * torch.imag(torch.conj(a) * b)).item(),
        (torch.abs(a) ** 2 - torch.abs(b) ** 2).item(),
    ])


def _qsp_simulate_bloch(phi, delta_val, Delta_0_ang, Omega_ang):
    """Simulate QSP sequence step-by-step for a single detuning value."""
    psi = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    bloch_vecs = [_spinor_to_bloch(psi)]
    pulse_info = []

    theta_val = delta_to_theta(delta_val, Delta_0_ang)
    H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2.0)
    W_mat = signal_operator(torch.tensor([theta_val], dtype=torch.float64))[0].to(torch.complex128)
    W_mat = H @ W_mat @ H
    Omega = Omega_ang
    norm = math.sqrt(Omega ** 2 + delta_val ** 2)

    for j, pv in enumerate(phi.tolist()):
        lamb = pv * norm / (2.0 * Omega)
        c, s = math.cos(lamb), math.sin(lamb)
        U_rx = torch.zeros((2, 2), dtype=torch.complex128)
        U_rx[0, 0] = complex(c, -s * delta_val / norm)
        U_rx[0, 1] = complex(0.0, -s * Omega / norm)
        U_rx[1, 0] = complex(0.0, -s * Omega / norm)
        U_rx[1, 1] = complex(c, +s * delta_val / norm)
        psi = U_rx @ psi
        bloch_vecs.append(_spinor_to_bloch(psi))
        pulse_info.append((0, pv, abs(pv)))

        if j < len(phi) - 1:
            psi = W_mat @ psi
            bloch_vecs.append(_spinor_to_bloch(psi))
            tau_W = get_wait_time(Delta_0_ang)
            pulse_info.append((0, 0.0, tau_W * Omega))

    return np.array(bloch_vecs), pulse_info


def nn_bloch_animation(phi, Delta_0_ang, Omega_ang, delta_vals_ang,
                        alpha_vals_rad, sigma_ang, out_path):
    """Bloch animation for 3*N qubits (N peaks x 3 triplets: delta, delta±sigma)."""
    if not _HAS_QUTIP:
        return None, "qutip is not installed — Bloch animation unavailable."

    sim_deltas = []
    for dv in delta_vals_ang:
        sim_deltas.extend([dv - sigma_ang, dv, dv + sigma_ang])
    sim_deltas = [max(-Delta_0_ang, min(Delta_0_ang, d)) for d in sim_deltas]

    delta_MHz_labels = [d / (2 * math.pi) for d in sim_deltas]
    epsilon_list = [0.0] * len(sim_deltas)

    bloch_list, pinfo_list, fid_list = [], [], []
    for dv in sim_deltas:
        bvecs, pinfo = _qsp_simulate_bloch(phi, dv, Delta_0_ang, Omega_ang)
        bloch_list.append(bvecs)
        pinfo_list.append(pinfo)

        ni = np.argmin([abs(dv - d) for d in delta_vals_ang])
        alpha_i = alpha_vals_rad[ni]
        F = _gate_fidelity_batch(
            phi, torch.tensor([dv], dtype=torch.float64),
            alpha_i, Delta_0_ang, Omega_ang,
        )
        fid_list.append(F[0].item())

    label_list = []
    for idx, d_MHz in enumerate(delta_MHz_labels):
        g = idx // 3
        av = alpha_vals_rad[g]
        target_str = f"Rx({av / math.pi:.3f}\u03c0)"
        label_list.append(
            f"delta = {d_MHz:.1f} MHz, F = {fid_list[idx]:.4f}, target = {target_str}"
        )

    N = len(delta_vals_ang)
    sigma_MHz = sigma_ang / (2 * math.pi)
    title = (
        f"NN QSP Ensemble Bloch — {N} peaks × 3 qubits "
        f"(δ, δ±σ={sigma_MHz:.1f} MHz)"
    )
    animate_multi_error_bloch(
        bloch_list, pinfo_list, fid_list,
        delta_MHz_labels, epsilon_list,
        name=title,
        save_path=out_path,
        phase_only=True,
        Omega=Omega_ang / (2 * math.pi),
        label_list=label_list,
    )
    return out_path, None


# ──────────────────────────────────────────────────────────────────────────────
#  Input parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_float_list(raw: str) -> Tuple[List[float], str]:
    cleaned = raw.replace("\n", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return [], "Enter a comma-separated list of numbers."
    values = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            return [], f"Could not parse '{p}' as a number."
    return values, ""


# ──────────────────────────────────────────────────────────────────────────────
#  Page layout
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="NN Controlled Unitary Demo", layout="wide")
st.title("Controlled Unitary Demo via Neural Network")

description = r"""
This demo uses a **pretrained neural network** to instantly generate control
pulse sequences for implementing target X-rotations on a qubit using detuning
as the control parameter.

Unlike the gradient-based demo that trains QSP phases from scratch (~10 min),
the neural network performs **instant inference** (< 1 second) to produce
$R_x(\alpha_i)$ at each detuning peak $\delta_i$.

The detuning peaks are fixed at $\delta \in \{-100, -32, 32, 100\}$ MHz
(hBN sample configuration).
"""
st.markdown(description)

# ── Sidebar: Model configuration ─────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Configuration")
    K = st.number_input("K (QSP degree)", min_value=1, value=DEFAULT_K, step=1)
    Omega_max_mhz = st.number_input("Omega_max (MHz)", min_value=0.0,
                                     value=DEFAULT_OMEGA_MHZ, step=1.0)
    Delta_0_mhz = DELTA_0_MHZ
    robustness_window_mhz = ROBUSTNESS_WINDOW_MHZ

with col2:
    st.subheader("Output Settings")
    out_dir = st.text_input("out_dir", value="nn_plots")

st.subheader("Target Values")
st.markdown(
    f"**Fixed detuning peaks:** {DELTA_CENTERS_MHZ} MHz  \n"
    f"**Delta_0:** {DELTA_0_MHZ} MHz | "
    f"**Robustness window:** {ROBUSTNESS_WINDOW_MHZ} MHz"
)
alpha_default = "0.3333, 1, 0.5, 1.25"
alpha_list_str = st.text_area(
    "alpha_vals (radians in π units, length 4)", value=alpha_default, height=80
)

run_btn = st.button("Run Inference")

# ── Training section (if no weights) ─────────────────────────────────────────

weight_path = get_weight_path("neural_network_optimization", Omega_max_mhz, int(K))
weights_exist = os.path.exists(weight_path)

if not weights_exist:
    st.warning(
        f"No pretrained weights found at `{weight_path}`.  \n"
        f"Train a model first, or use the button below."
    )
    train_steps = st.number_input("Training steps", min_value=100, value=10000, step=100)
    if st.button("Train Model"):
        with st.spinner(f"Training NN (Omega={Omega_max_mhz}, K={K})..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def progress_cb(step, total, loss, eta):
                progress_bar.progress(step / total)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                progress_text.write(
                    f"Step {step}/{total} — loss {loss:.3e} — ETA {eta_min:02d}:{eta_sec:02d}"
                )

            train_model(
                Omega_mhz=Omega_max_mhz,
                K=int(K),
                steps=int(train_steps),
                verbose=False,
                progress_cb=progress_cb,
            )
        st.success(f"Training complete. Weights saved to `{weight_path}`.")
        weights_exist = True
        st.rerun()


if "results" not in st.session_state:
    st.session_state["results"] = None
if "viz" not in st.session_state:
    st.session_state["viz"] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────────────────────

if run_btn and weights_exist:
    alpha_val_scaled, alpha_err = parse_float_list(alpha_list_str)

    errors = []
    if alpha_err:
        errors.append(f"alpha_vals: {alpha_err}")
    if not errors and len(alpha_val_scaled) != N_PEAKS:
        errors.append(
            f"alpha_vals length is {len(alpha_val_scaled)}, expected {N_PEAKS}."
        )

    if errors:
        for e in errors:
            st.error(e)
    else:
        os.makedirs(out_dir, exist_ok=True)
        torch.set_default_dtype(torch.float64)

        # Load model and run inference
        model = load_model(Omega_mhz=Omega_max_mhz, K=int(K))
        alpha_vals_rad = [a * math.pi for a in alpha_val_scaled]
        delta_vals_ang = [2 * math.pi * d for d in DELTA_CENTERS_MHZ]

        phi = generate_phi(model, alpha_vals_rad)
        pulse_df = phi_to_pulse_df(phi, model.Omega_mhz, model.Delta_0_mhz)

        # Compute fidelity
        fid = compute_fidelity(model, alpha_vals_rad)
        runtime = compute_runtime(model, alpha_vals_rad)

        # Matrix element plot
        plot_path = os.path.join(out_dir, f"nn_matrix_elements_K{int(K)}.png")
        nn_matrix_element_plot(
            phi, model.Delta_0_ang, model.Omega_ang,
            delta_vals_ang, alpha_vals_rad,
            model.robustness_window_ang, plot_path,
        )

        st.session_state["results"] = {
            "pulse_df": pulse_df,
            "phi": phi,
            "delta_vals_ang": delta_vals_ang,
            "alpha_vals_rad": alpha_vals_rad,
            "fidelity": fid,
            "runtime": runtime,
            "plot_path": plot_path,
            "model": model,
            "args": {
                "K": int(K),
                "Omega_max": float(Omega_max_mhz),
                "Delta_0": float(Delta_0_mhz),
                "robustness_window": float(robustness_window_mhz),
                "out_dir": out_dir,
            },
        }
        st.session_state["viz"] = {}


# ──────────────────────────────────────────────────────────────────────────────
#  Results & Visualizations
# ──────────────────────────────────────────────────────────────────────────────

if st.session_state["results"] is not None:
    results = st.session_state["results"]
    viz = st.session_state["viz"]

    st.success(
        f"Inference complete. "
        f"Gate Fidelity: {results['fidelity']:.6f} | "
        f"Runtime: {results['runtime']*1e3:.1f} ns"
    )

    phi = results["phi"]
    delta_vals_ang = results["delta_vals_ang"]
    alpha_vals_rad = results["alpha_vals_rad"]
    model = results["model"]
    args = results["args"]
    out_dir_stored = args["out_dir"]
    os.makedirs(out_dir_stored, exist_ok=True)

    target_labels = [
        f"Target {i}: δ={delta_vals_ang[i]/(2*math.pi):.1f} MHz, "
        f"Rx({alpha_vals_rad[i]/math.pi:.3f}π)"
        for i in range(N_PEAKS)
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
            file_name="nn_pulse_schedule.csv",
            mime="text/csv",
            key="download_pulse_csv",
        )

        pulse_param_path = os.path.join(out_dir_stored, "nn_pulse_param_plot.png")
        if "pulse_param_path" not in viz or not os.path.exists(pulse_param_path):
            with st.spinner("Generating pulse schedule plot..."):
                plot_pulse_param(
                    out_dir_stored,
                    f"NN_K{args['K']}",
                    results["pulse_df"],
                    args["Omega_max"],
                )
                generated = os.path.join(out_dir_stored, f"NN_K{args['K']}.png")
                if os.path.exists(generated):
                    os.rename(generated, pulse_param_path)
            viz["pulse_param_path"] = pulse_param_path
            st.session_state["viz"] = viz

        if os.path.exists(pulse_param_path):
            st.image(pulse_param_path,
                     caption="Pulse schedule (Ωₓ, Ωᵧ, Ω_z vs time)",
                     use_column_width=True)

    # ── Tab 2: Matrix Element vs δ ────────────────────────────────────────────
    with tab_matrix:
        st.subheader("Matrix Element vs Detuning")
        if os.path.exists(results["plot_path"]):
            st.image(results["plot_path"],
                     caption="Re(u₀₀) and Im(u₀₁) vs δ with target windows",
                     use_column_width=True)

    # ── Tab 3: Fidelity Contour ───────────────────────────────────────────────
    with tab_contour:
        st.subheader("Fidelity Contour Plot")
        st.markdown(
            "Gate fidelity over (detuning δ, calibration error ε) for one target peak."
        )
        sel_c = st.selectbox("Select target", target_labels, key="sel_contour")
        idx_c = target_labels.index(sel_c)

        if st.button("Generate Fidelity Contour", key="btn_contour"):
            contour_path = os.path.join(out_dir_stored,
                                        f"nn_fidelity_contour_target{idx_c}.png")
            with st.spinner("Computing fidelity contour..."):
                nn_fidelity_contour_plot(
                    phi, model.Delta_0_ang, model.Omega_ang,
                    delta_vals_ang[idx_c], alpha_vals_rad[idx_c],
                    model.robustness_window_ang, contour_path,
                )
            viz[f"contour_{idx_c}"] = contour_path
            st.session_state["viz"] = viz

        cache_key = f"contour_{idx_c}"
        if cache_key in viz and os.path.exists(viz[cache_key]):
            st.image(viz[cache_key], caption="Fidelity contour",
                     use_column_width=True)

    # ── Tab 4: Fidelity vs Std ────────────────────────────────────────────────
    with tab_fid_std:
        st.subheader("Fidelity vs δ Spread")
        sel_f = st.selectbox("Select target", target_labels, key="sel_fid_std")
        idx_f = target_labels.index(sel_f)
        M_fid = st.number_input("Monte-Carlo samples M", min_value=100,
                                 value=2000, step=100, key="M_fid")

        if st.button("Generate Fidelity vs Std", key="btn_fid_std"):
            fid_std_path = os.path.join(out_dir_stored,
                                        f"nn_fidelity_by_std_target{idx_f}.png")
            with st.spinner("Computing fidelity vs std..."):
                nn_fidelity_by_std_plot(
                    phi, model.Delta_0_ang, model.Omega_ang,
                    delta_vals_ang[idx_f], alpha_vals_rad[idx_f],
                    fid_std_path, M=int(M_fid),
                )
            viz[f"fid_std_{idx_f}"] = fid_std_path
            st.session_state["viz"] = viz

        cache_key_f = f"fid_std_{idx_f}"
        if cache_key_f in viz and os.path.exists(viz[cache_key_f]):
            st.image(viz[cache_key_f], caption="Fidelity vs std(δ)",
                     use_column_width=True)

    # ── Tab 5: Bloch Animation ────────────────────────────────────────────────
    with tab_bloch:
        st.subheader("Ensemble Bloch Sphere Animation")
        sigma_MHz = model.robustness_window_ang / (2 * math.pi)
        st.markdown(
            f"Qubit-state trajectories for **{3 * N_PEAKS} qubits** "
            f"({N_PEAKS} peaks × 3): δᵢ − σ, δᵢ, δᵢ + σ with σ = {sigma_MHz:.1f} MHz."
        )
        if not _HAS_QUTIP:
            st.warning("qutip is not installed. Install with `pip install qutip`.")
        else:
            if st.button("Generate Bloch Animation", key="btn_bloch"):
                anim_path = os.path.join(out_dir_stored, "nn_bloch_animation.mp4")
                with st.spinner(
                    f"Simulating {3 * N_PEAKS} qubit trajectories..."
                ):
                    out, err_msg = nn_bloch_animation(
                        phi, model.Delta_0_ang, model.Omega_ang,
                        delta_vals_ang, alpha_vals_rad,
                        model.robustness_window_ang, anim_path,
                    )
                if err_msg:
                    st.error(err_msg)
                elif out and os.path.exists(out):
                    viz["bloch_path"] = out
                    st.session_state["viz"] = viz

            if "bloch_path" in viz and os.path.exists(viz["bloch_path"]):
                st.video(viz["bloch_path"])
                with open(viz["bloch_path"], "rb") as vf:
                    st.download_button(
                        label="Download animation (mp4)",
                        data=vf.read(),
                        file_name="nn_bloch_animation.mp4",
                        mime="video/mp4",
                        key="download_bloch",
                    )
