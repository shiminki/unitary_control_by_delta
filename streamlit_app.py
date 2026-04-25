"""Streamlit demo: load a trained PulseNet and visualise its output.

This app does NOT train.  If the (Omega, K) combination has no weights yet,
run:

    python -m neural_network_optimization.scaling_law

to populate ``neural_network_optimization/weights/``.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from neural_network_optimization.constants import (
    DELTA_CENTERS_MHZ,
    K_LIST,
    OMEGA_LIST,
)
from neural_network_optimization.inference import (
    compute_fidelity,
    compute_runtime,
    generate_phi,
    generate_pulse,
    load_model,
)
from neural_network_optimization.model import get_weight_path
from neural_network_optimization.pca_analysis import (
    format_analytical_form,
    run_pca,
)
from neural_network_optimization.plotting import (
    plot_amplitude_components,
    plot_comparative_matrix_elements,
    plot_comparative_pulses,
    plot_matrix_element,
    plot_pca_overview,
)

WEIGHT_DIR = "neural_network_optimization/weights"
CACHE_DIR = "outputs/streamlit_cache"

st.set_page_config(page_title="QSP Pulse Demo", layout="wide")


@st.cache_resource(show_spinner="Loading weights …")
def _load_model(omega, K):
    return load_model(Omega=omega, K=K, weight_dir=WEIGHT_DIR, device="cpu")


@st.cache_data(show_spinner="Running PCA …")
def _cached_pca(omega, K, peak_index, n_samples):
    model = _load_model(omega, K)
    return run_pca(model, peak_index, n_samples=n_samples, save_figures=False)


def _pulse_plot(pulse_df, omega_mhz, delta_0_mhz):
    ts = pulse_df["t (us)"].to_numpy()
    hx = pulse_df["Omega_x (2pi MHz)"].to_numpy()
    hz = pulse_df["Omega_z (2pi MHz)"].to_numpy()
    edges = np.concatenate(([0.0], np.cumsum(ts)))
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].step(edges, np.r_[hx, hx[-1]], where="post", color="C0")
    ax[0].set_ylabel(r"$\Omega_x$ (2π MHz)")
    ax[0].set_ylim(-1.1 * omega_mhz, 1.1 * omega_mhz)
    ax[0].grid(alpha=0.3)
    ax[1].step(edges, np.r_[hz, hz[-1]], where="post", color="C1")
    ax[1].set_ylabel(r"$\Omega_z$ (2π MHz)")
    ax[1].set_xlabel("Time (μs)")
    ax[1].set_ylim(-5, delta_0_mhz + 10)
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _phi_bar(phi):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(phi)), phi, color="steelblue")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("QSP phase index j")
    ax.set_ylabel(r"$\phi_j$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    st.title("Detuning-selective QSP pulse generator")
    st.caption(
        "Load pre-trained weights, pick four rotation angles, and inspect the "
        "generated pulse schedule / matrix element profile / fidelity / "
        "PCA decomposition."
    )

    with st.sidebar:
        st.header("Model")
        omega = st.selectbox("Ω (MHz)", OMEGA_LIST, index=len(OMEGA_LIST) - 1)
        K = st.selectbox("K", K_LIST, index=1)
        path = get_weight_path(WEIGHT_DIR, omega, K)
        if not os.path.exists(path):
            st.error(
                f"No weights at `{path}`.\n\n"
                "Run `python -m neural_network_optimization.scaling_law` to train "
                "all 12 configurations, or "
                f"`python -m neural_network_optimization.train --Omega {omega} --K {K}` "
                "for this single one."
            )
            return

        model = _load_model(omega, K)
        st.markdown("**Peaks**")
        st.write(f"Δ centres (MHz): `{DELTA_CENTERS_MHZ}`")
        st.write(f"σ = ±{model.robustness_window_mhz} MHz, Δ₀ = {model.Delta_0_mhz} MHz")

    st.subheader("Rotation angles α₀..α₃ (rad)")
    cols = st.columns(model.N_peaks)
    alphas = []
    defaults = [math.pi / 2, math.pi, math.pi / 3, 0.0]
    for i, col in enumerate(cols):
        alphas.append(col.slider(
            f"α_{i}", min_value=0.0, max_value=4 * math.pi,
            value=float(defaults[i % len(defaults)]), step=0.05,
            format="%.3f",
        ))

    phi = generate_phi(model, alphas)
    pulse_df = generate_pulse(model, alphas)
    runtime_us = compute_runtime(model, alphas)

    t_pulse, t_matrix, t_fid, t_phi, t_pca = st.tabs([
        "Pulse", "Matrix element", "Fidelity", "φ vector", "PCA",
    ])

    with t_pulse:
        st.pyplot(_pulse_plot(pulse_df, model.Omega_mhz, model.Delta_0_mhz))
        st.dataframe(pulse_df.style.format("{:.5f}"))

    with t_matrix:
        os.makedirs(CACHE_DIR, exist_ok=True)
        img_path = os.path.join(CACHE_DIR, f"matrix_Omega{omega}_K{K}.png")
        plot_matrix_element(model, alphas, img_path, sample_size_for_fidelity=2000)
        st.image(img_path)

    with t_fid:
        with st.spinner("Evaluating fidelity …"):
            fid = compute_fidelity(model, alphas, sample_size=3000)
        st.metric("Fidelity", f"{fid:.6f}")
        st.metric("Infidelity (1 − F)", f"{1 - fid:.2e}")
        st.metric("Runtime (μs)", f"{runtime_us:.4f}")

    with t_phi:
        st.pyplot(_phi_bar(phi.numpy()))

    with t_pca:
        peak = st.selectbox("Peak index", list(range(model.N_peaks)), index=0)
        n_samples = st.slider("Samples along α", 64, 1024, 256, step=64)
        pca_result = _cached_pca(omega, K, peak, n_samples)

        os.makedirs(CACHE_DIR, exist_ok=True)
        overview_path = os.path.join(CACHE_DIR, f"pca_overview_Ω{omega}_K{K}_p{peak}.png")
        amp_path = os.path.join(CACHE_DIR, f"pca_amp_Ω{omega}_K{K}_p{peak}.png")
        cp_path = os.path.join(CACHE_DIR, f"comp_pulses_Ω{omega}_K{K}_p{peak}.png")
        cm_path = os.path.join(CACHE_DIR, f"comp_me_Ω{omega}_K{K}_p{peak}.png")
        plot_pca_overview(pca_result, peak, overview_path)
        plot_amplitude_components(pca_result, peak, amp_path)
        plot_comparative_pulses(model, pca_result, peak, cp_path)
        plot_comparative_matrix_elements(
            model, pca_result, peak, cm_path, sample_size_for_fidelity=1000,
        )
        st.image(overview_path)
        st.image(amp_path)
        st.image(cp_path)
        st.image(cm_path)
        st.text(format_analytical_form(pca_result))


if __name__ == "__main__":
    main()
