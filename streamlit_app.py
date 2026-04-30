"""Streamlit demo: load a trained QSPPhaseNet and visualise its output.

Weights are loaded from  scaling_law_output/nn_data/Omega{Omega}_K{K}/model_final.pt

To train weights, run:
    python scaling_law.py --skip_classical true
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from neural_network_optimization_QSP import QSPPhaseNet, predict_phi
from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    TrainConfig,
    fidelity,
    get_control_runtime,
    get_wait_time,
    plot_matrix_element_vs_delta,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Fixed physical parameters (hBN sample)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_DIR      = "scaling_law_output/nn_data"
CACHE_DIR       = "outputs/streamlit_cache"

DELTA_VALS_MHZ  = [-100.0, -32.0, 32.0, 100.0]   # peak centres (MHz)
DELTA_0_MHZ     = 200.0                            # max detuning range (MHz)
SIGMA_MHZ       = 10.0                             # robustness half-width (MHz)
N_PEAKS         = 4

OMEGA_LIST      = [40, 80, 120, 160]   # MHz
K_LIST          = [50, 70, 100]

PCA_M           = 200   # number of alpha steps; alphas run 0 … 4π in steps of 2π/M


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weight_path(omega_mhz: int, K: int) -> str:
    return os.path.join(WEIGHT_DIR, f"Omega{omega_mhz}_K{K}", "model_final.pt")


def _make_cfg(omega_mhz: int, K: int) -> TrainConfig:
    return TrainConfig(
        Omega_max=2 * math.pi * omega_mhz,
        Delta_0=2 * math.pi * DELTA_0_MHZ,
        robustness_window=2 * math.pi * SIGMA_MHZ,
        K=K,
        device="cpu",
    )


@st.cache_resource(show_spinner="Loading weights …")
def _load_model(omega_mhz: int, K: int) -> QSPPhaseNet:
    model = QSPPhaseNet(N=N_PEAKS, K=K)
    path = _weight_path(omega_mhz, K)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval().double()
    return model


def _phi_from_alphas(model: QSPPhaseNet, alphas: list[float]) -> torch.Tensor:
    """Predict QSP phase vector for a list of N rotation angles."""
    a = torch.tensor(alphas, dtype=torch.float64)
    return predict_phi(model, a, device="cpu")


def _phi_to_pulse_df(phi: torch.Tensor, cfg: TrainConfig, omega_mhz: float) -> pd.DataFrame:
    """Convert a QSP phase vector to a pulse schedule DataFrame."""
    tau_us = get_wait_time(cfg.Delta_0)           # signal wait time
    delta_0_mhz = cfg.Delta_0 / (2 * math.pi)

    t_rows, hx_rows, hz_rows = [], [], []
    for j, p in enumerate(phi.tolist()):
        t_rows.append(abs(p) / cfg.Omega_max)
        hx_rows.append(omega_mhz * math.copysign(1.0, p))
        hz_rows.append(0.0)
        if j < len(phi) - 1:
            t_rows.append(tau_us)
            hx_rows.append(0.0)
            hz_rows.append(delta_0_mhz)

    return pd.DataFrame({
        "t (us)":           t_rows,
        "Omega_x (2pi MHz)": hx_rows,
        "Omega_z (2pi MHz)": hz_rows,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  PCA analysis:  phi(α) for α ∈ {0, 2π/M, 4π/M, …, 4π}, peak 0 only
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running PCA …")
def _run_pca(omega_mhz: int, K: int, M: int = PCA_M):
    """
    Build matrix  Φ = [φ(0) | φ(2π/M) | … | φ(4π)]  (shape K+1 × 2M+1)
    where φ(α) = predict_phi(model, [α, 0, 0, 0]).

    Returns
    -------
    alphas        : (2M+1,) alpha values in radians
    Phi           : (2M+1, K+1) phase matrix (raw, not centred)
    components    : (n_comp, K+1) principal directions (rows of Vt)
    scores        : (2M+1, n_comp) amplitudes along each PC
    explained_var : (n_comp,) fraction of variance per component
    """
    model = _load_model(omega_mhz, K)

    alphas = np.linspace(0.0, 4 * math.pi, 2 * M + 1)   # 401 values
    Phi = np.zeros((len(alphas), K + 1))

    for i, a in enumerate(alphas):
        alpha_vec = torch.tensor([a, 0.0, 0.0, 0.0], dtype=torch.float64)
        with torch.no_grad():
            phi = predict_phi(model, alpha_vec, device="cpu").numpy()
        Phi[i] = phi

    # Centre columns (mean over α)
    Phi_c = Phi - Phi.mean(axis=0, keepdims=True)

    # Thin SVD  →  Phi_c = U S Vt
    U, s, Vt = np.linalg.svd(Phi_c, full_matrices=False)   # U:(2M+1, r), s:(r,), Vt:(r, K+1)
    explained_var = s ** 2 / (s ** 2).sum()

    scores     = U * s          # (2M+1, r)  amplitudes per PC
    components = Vt             # (r, K+1)   principal directions

    return alphas, Phi, components, scores, explained_var


# ─────────────────────────────────────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pulse(phi: torch.Tensor, cfg: TrainConfig, omega_mhz: float):
    pulse_df = _phi_to_pulse_df(phi, cfg, omega_mhz)
    ts  = pulse_df["t (us)"].to_numpy()
    hx  = pulse_df["Omega_x (2pi MHz)"].to_numpy()
    hz  = pulse_df["Omega_z (2pi MHz)"].to_numpy()
    edges = np.concatenate(([0.0], np.cumsum(ts)))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax0.step(edges, np.r_[hx, hx[-1]], where="post", color="C0")
    ax0.set_ylabel(r"$\Omega_x$ (2π MHz)")
    ax0.set_ylim(-1.1 * omega_mhz, 1.1 * omega_mhz)
    ax0.grid(alpha=0.3)
    ax1.step(edges, np.r_[hz, hz[-1]], where="post", color="C1")
    ax1.set_ylabel(r"$\Omega_z$ (2π MHz)")
    ax1.set_xlabel("Time (μs)")
    ax1.set_ylim(-5, DELTA_0_MHZ + 10)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    return fig, pulse_df


def _plot_phi_bar(phi: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(phi)), phi, color="steelblue", width=1.0)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Phase index j")
    ax.set_ylabel(r"$\phi_j$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_pca_scree(explained_var: np.ndarray, n_show: int = 20):
    n = min(n_show, len(explained_var))
    fig, ax = plt.subplots(figsize=(7, 4))
    cumvar = np.cumsum(explained_var[:n])
    ax.bar(np.arange(1, n + 1), explained_var[:n] * 100, color="steelblue", label="Individual")
    ax.step(np.arange(1, n + 1), cumvar * 100, where="mid", color="C1", lw=2, label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot  –  φ(α) at peak 0  (δ = −100 MHz)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_pca_scores(alphas: np.ndarray, scores: np.ndarray, n_comp: int = 4):
    fig, axes = plt.subplots(n_comp, 1, figsize=(10, 2.5 * n_comp), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(alphas / math.pi, scores[:, i], lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"Score PC{i + 1}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("α  (units of π)")
    axes[0].set_title("PCA Score Amplitudes  –  φ(α) at peak 0  (δ = −100 MHz)")
    plt.tight_layout()
    return fig


def _plot_pca_components(components: np.ndarray, n_comp: int = 4):
    K_plus_1 = components.shape[1]
    fig, axes = plt.subplots(n_comp, 1, figsize=(10, 2.5 * n_comp), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(K_plus_1), components[i], lw=1.0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"PC{i + 1}")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Phase index j")
    axes[0].set_title("Principal Component Vectors  φ-space")
    plt.tight_layout()
    return fig


def _plot_phi_vs_alpha(alphas: np.ndarray, Phi: np.ndarray, n_phases: int = 10):
    """Show a few individual φ_j entries as a function of α."""
    K_plus_1 = Phi.shape[1]
    indices = np.linspace(0, K_plus_1 - 1, n_phases, dtype=int)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j in indices:
        ax.plot(alphas / math.pi, Phi[:, j], lw=1.0, label=f"j={j}")
    ax.set_xlabel("α  (units of π)")
    ax.set_ylabel(r"$\phi_j(\alpha)$")
    ax.set_title("Selected phase components φ_j(α) at peak 0")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="QSP Pulse Demo", layout="wide")
    st.title("Detuning-selective QSP pulse generator")
    st.caption(
        "Load pre-trained QSPPhaseNet weights, pick four rotation angles, "
        "and inspect the generated pulse schedule / matrix element / fidelity / "
        "φ vector / PCA decomposition."
    )

    # ── Sidebar: model selection ──────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        omega = st.selectbox("Ω (MHz)", OMEGA_LIST, index=len(OMEGA_LIST) - 1)
        K = st.selectbox("K", K_LIST, index=1)

        path = _weight_path(omega, K)
        if not os.path.exists(path):
            st.error(
                f"No weights at `{path}`.\n\n"
                "Run `python scaling_law.py --skip_classical true` to train all models."
            )
            return

        model = _load_model(omega, K)
        cfg   = _make_cfg(omega, K)

        st.markdown("**Fixed peaks**")
        st.write(f"δ centres (MHz): `{DELTA_VALS_MHZ}`")
        st.write(f"σ = ±{SIGMA_MHZ} MHz,  Δ₀ = {DELTA_0_MHZ} MHz")

    # ── Rotation angles ───────────────────────────────────────────────────────
    st.subheader("Rotation angles α₀ … α₃ (rad)")
    cols    = st.columns(N_PEAKS)
    defaults = [math.pi / 2, math.pi, math.pi / 3, 0.0]
    alphas  = [
        col.slider(
            f"α_{i}  (peak δ={DELTA_VALS_MHZ[i]} MHz)",
            min_value=0.0, max_value=4 * math.pi,
            value=float(defaults[i]), step=0.05, format="%.3f",
        )
        for i, col in enumerate(cols)
    ]

    phi = _phi_from_alphas(model, alphas)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t_pulse, t_matrix, t_fid, t_phi, t_pca = st.tabs([
        "Pulse", "Matrix element", "Fidelity", "φ vector", "PCA",
    ])

    delta_vals_t = torch.tensor(
        [2 * math.pi * d for d in DELTA_VALS_MHZ], dtype=torch.float64
    )
    alpha_vals_t = torch.tensor(alphas, dtype=torch.float64)

    with t_pulse:
        fig, pulse_df = _plot_pulse(phi, cfg, omega)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(pulse_df.style.format("{:.6f}"))

    with t_matrix:
        os.makedirs(CACHE_DIR, exist_ok=True)
        img_path = os.path.join(CACHE_DIR, f"matrix_Omega{omega}_K{K}.png")
        with st.spinner("Computing matrix elements …"):
            plot_matrix_element_vs_delta(phi, cfg, delta_vals_t, alpha_vals_t, img_path)
        st.image(img_path)

    with t_fid:
        with st.spinner("Evaluating fidelity …"):
            fid     = fidelity(phi, delta_vals_t, alpha_vals_t, cfg)
            runtime = get_control_runtime(phi, cfg)
        st.metric("Fidelity", f"{fid:.6f}")
        st.metric("Infidelity (1 − F)", f"{1 - fid:.2e}")
        st.metric("Runtime (μs)", f"{runtime:.4f}")

    with t_phi:
        st.pyplot(_plot_phi_bar(phi.numpy()))
        plt.close("all")

    with t_pca:
        _show_pca_tab(omega, K)


def _show_pca_tab(omega_mhz: int, K: int):
    st.markdown(
        r"""
**PCA of φ(α) at peak 0 (δ = −100 MHz)**

φ(α) is the QSP phase vector predicted for R_x(α) at the leftmost peak,
with all other peaks set to identity (α_1 = α_2 = α_3 = 0).

The matrix **Φ** = [φ(0), φ(2π/M), φ(4π/M), …, φ(4π)] with **M = 200**
is assembled (401 columns), mean-centred, and decomposed by SVD.
"""
    )

    with st.spinner("Running PCA …"):
        alphas, Phi, components, scores, explained_var = _run_pca(omega_mhz, K, M=PCA_M)

    n_comp_show = st.slider("Number of PCs to display", 1, min(10, len(explained_var)), 4)

    col_left, col_right = st.columns(2)
    with col_left:
        fig = _plot_pca_scree(explained_var, n_show=20)
        st.pyplot(fig)
        plt.close(fig)

        ev_df = pd.DataFrame({
            "PC":               np.arange(1, len(explained_var) + 1),
            "Explained Var (%)": explained_var * 100,
            "Cumulative (%)":    np.cumsum(explained_var) * 100,
        })
        st.dataframe(ev_df.head(20).style.format("{:.3f}", subset=["Explained Var (%)", "Cumulative (%)"]))

    with col_right:
        fig = _plot_phi_vs_alpha(alphas, Phi)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("PC Score Amplitudes  (how much each PC contributes at each α)")
    fig = _plot_pca_scores(alphas, scores, n_comp=n_comp_show)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Principal Component Vectors  (structure in φ-space)")
    fig = _plot_pca_components(components, n_comp=n_comp_show)
    st.pyplot(fig)
    plt.close(fig)

    # Reconstruction check for a single alpha
    st.subheader("Reconstruction check")
    alpha_check = st.slider(
        "α for reconstruction (rad)", 0.0, float(4 * math.pi),
        value=float(math.pi), step=0.05, format="%.3f",
    )
    n_comp_recon = st.slider("PCs used for reconstruction", 1, min(20, scores.shape[1]), 5)

    idx   = int(np.argmin(np.abs(alphas - alpha_check)))
    mean_phi = Phi.mean(axis=0)
    phi_true = Phi[idx]
    phi_recon = mean_phi + (scores[idx, :n_comp_recon] @ components[:n_comp_recon])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(phi_true,  lw=1.2, label=f"φ(α={alpha_check/math.pi:.2f}π)  [true]")
    ax.plot(phi_recon, lw=1.2, linestyle="--", label=f"Reconstruction  (top {n_comp_recon} PCs)")
    ax.set_xlabel("Phase index j")
    ax.set_ylabel(r"$\phi_j$")
    ax.set_title(
        f"Reconstruction with top {n_comp_recon} PCs  "
        f"({np.cumsum(explained_var)[n_comp_recon - 1]*100:.1f}% variance)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
