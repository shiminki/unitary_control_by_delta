"""Streamlit demo: load a trained QSPPhaseNet and visualise its output.

Weights are loaded from  scaling_law_output/nn_data/Omega{Omega}_K{K}/model_final.pt

To train weights, run:
    python scaling_law.py --skip_classical true
"""

import hashlib
import json
import math
import os
import time

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
    train,
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
#  Input parsing (single-pulse mode)
# ─────────────────────────────────────────────────────────────────────────────

def parse_float_list(raw: str):
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


# ─────────────────────────────────────────────────────────────────────────────
#  Phase cache helpers (single-pulse mode)
# ─────────────────────────────────────────────────────────────────────────────

def _make_cache_key(
    delta_val_scaled,
    alpha_val_scaled,
    K: int,
    Delta_0_MHz: float,
    Omega_max_MHz: float,
    robustness_window_MHz: float,
) -> str:
    payload = json.dumps(
        {
            "delta_vals": [round(v, 8) for v in delta_val_scaled],
            "alpha_vals": [round(v, 8) for v in alpha_val_scaled],
            "K": int(K),
            "Delta_0_MHz": round(Delta_0_MHz, 6),
            "Omega_max_MHz": round(Omega_max_MHz, 6),
            "robustness_window_MHz": round(robustness_window_MHz, 6),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_csv_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"phi_{key}.csv")


def _load_phase_cache(cache_dir: str, key: str):
    path = _cache_csv_path(cache_dir, key)
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    phi = torch.tensor(df["phi"].values, dtype=torch.float64)
    final_loss = float(df["final_loss"].iloc[0]) if "final_loss" in df.columns else float("nan")
    return phi, final_loss


def _save_phase_cache(cache_dir: str, key: str, phi: torch.Tensor, final_loss: float) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_csv_path(cache_dir, key)
    loss_col = [float(final_loss)] + [float("nan")] * (len(phi) - 1)
    pd.DataFrame({
        "index": list(range(len(phi))),
        "phi": phi.numpy().tolist(),
        "final_loss": loss_col,
    }).to_csv(path, index=False)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  NN helpers
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


def _phi_from_alphas(model: QSPPhaseNet, alphas: list) -> torch.Tensor:
    a = torch.tensor(alphas, dtype=torch.float64)
    return predict_phi(model, a, device="cpu")


def _phi_to_pulse_df(phi: torch.Tensor, cfg: TrainConfig, omega_mhz: float) -> pd.DataFrame:
    tau_us = get_wait_time(cfg.Delta_0)
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
#  PCA analysis
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running PCA …")
def _run_pca(omega_mhz: int, K: int, M: int = PCA_M):
    model = _load_model(omega_mhz, K)

    alphas = np.linspace(0.0, 4 * math.pi, 2 * M + 1)
    Phi = np.zeros((len(alphas), K + 1))

    for i, a in enumerate(alphas):
        alpha_vec = torch.tensor([a, 0.0, 0.0, 0.0], dtype=torch.float64)
        with torch.no_grad():
            phi = predict_phi(model, alpha_vec, device="cpu").numpy()
        Phi[i] = phi

    Phi_c = Phi - Phi.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Phi_c, full_matrices=False)
    explained_var = s ** 2 / (s ** 2).sum()

    scores     = U * s
    components = Vt

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

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        mode = st.radio(
            "Solver",
            ["Single-Pulse Optimization", "Neural Network"],
            help=(
                "**Neural Network**: fast inference from pre-trained QSPPhaseNet weights.\n\n"
                "**Single-Pulse Optimization**: train QSP phases from scratch via gradient descent."
            ),
        )
        st.divider()

        if mode == "Neural Network":
            st.header("Model")
            omega = st.selectbox("Ω (MHz)", OMEGA_LIST, index=1)
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

    # ── Neural-Network mode ───────────────────────────────────────────────────

    if mode == "Neural Network":
        st.caption(
            "Load pre-trained QSPPhaseNet weights, pick four rotation angles, "
            "and inspect the generated pulse schedule / matrix element / fidelity / "
            "φ vector / PCA decomposition."
        )

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

    # ── Single-Pulse Optimization mode ────────────────────────────────────────

    else:
        st.caption(
            "Train QSP phases from scratch via gradient descent. "
            "Provide target detunings and rotation angles, then click **Run Training**."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Core Arguments")
            K = st.number_input("K (max phase index)", min_value=1, value=70, step=1)
            N = st.number_input("N (num peaks)", min_value=1, value=4, step=1)
            Delta_0_mhz = st.number_input("Delta_0 (MHz)", min_value=0.0, value=200.0, step=1.0)
            robustness_window_mhz = st.number_input("robustness_window (MHz)", min_value=0.0, value=10.0, step=0.1)
            Omega_max_mhz = st.number_input("Omega_max (MHz)", min_value=0.0, value=80.0, step=1.0)

        with col2:
            st.subheader("Other Arguments")
            steps = st.number_input("Training steps", min_value=1, value=8000, step=100)
            lr = st.number_input("lr", min_value=0.0, value=5e-2, step=1e-3, format="%.6f")
            default_device = "cuda" if torch.cuda.is_available() else "cpu"
            device = st.selectbox("device", options=["cpu", "cuda"],
                                  index=0 if default_device == "cpu" else 1)
            out_dir = st.text_input("out_dir", value="plots_relaxed")
            cache_dir = st.text_input(
                "cache_dir (leave blank to disable caching)",
                value="phase_cache",
            )

        st.subheader("Target Values")
        alpha_default = "0.3333, 1, 0.5, 1.25"
        delta_default = "-100, -32, 32, 100"
        alpha_list_scaled = st.text_area(
            "alpha_vals (units of π, length N)", value=alpha_default, height=80
        )
        delta_list_scaled = st.text_area(
            "delta_vals (MHz, length N)", value=delta_default, height=80
        )

        run_btn = st.button("Run Training")

        if "results" not in st.session_state:
            st.session_state["results"] = None

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

                cfg_sp = TrainConfig(
                    Omega_max=2 * math.pi * Omega_max_mhz,
                    Delta_0=2 * math.pi * Delta_0_mhz,
                    robustness_window=2 * math.pi * robustness_window_mhz,
                    K=int(K),
                    steps=int(steps),
                    lr=float(lr),
                    device=device,
                    out_dir=out_dir,
                )

                delta_vals = torch.tensor(delta_val_scaled, device=device) * (2 * math.pi)
                alpha_vals = torch.tensor(alpha_val_scaled, device=device) * math.pi

                if (delta_vals.abs() > cfg_sp.Delta_0).any():
                    st.warning("Some delta_vals exceed |Delta_0| after unit conversion.")

                # Cache check
                _use_cache = bool(cache_dir.strip())
                _cache_key = _make_cache_key(
                    delta_val_scaled, alpha_val_scaled,
                    int(K), float(Delta_0_mhz), float(Omega_max_mhz),
                    float(robustness_window_mhz),
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
                        progress_bar  = st.progress(0)
                        progress_text = st.empty()
                        _start_t      = time.perf_counter()

                        def progress_cb(step: int, total: int, loss: float, eta: float) -> None:
                            elapsed  = time.perf_counter() - _start_t
                            pct      = step / total
                            rate     = step / elapsed if elapsed > 0 else 0.0
                            bar_w    = 25
                            filled   = int(bar_w * pct)
                            bar      = "█" * filled + "░" * (bar_w - filled)
                            el_m, el_s  = int(elapsed // 60), int(elapsed % 60)
                            eta_m, eta_s = int(eta // 60), int(eta % 60)
                            progress_bar.progress(pct)
                            progress_text.code(
                                f"Training: {int(pct * 100):3d}%"
                                f"|{bar}|"
                                f" {step}/{total}"
                                f" [{el_m:02d}:{el_s:02d}<{eta_m:02d}:{eta_s:02d},"
                                f" {rate:5.1f}it/s,"
                                f" loss={loss:.3e}]"
                            )

                        phi_final, final_loss, _ = train(
                            cfg_sp,
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

                # Build pulse schedule
                tau_us = get_wait_time(cfg_sp.Delta_0)
                t_rows, hx_rows, hz_rows = [], [], []
                for i, phi_val in enumerate(phi_final.tolist()):
                    t_rows.append(np.abs(phi_val) / cfg_sp.Omega_max)
                    hx_rows.append(Omega_max_mhz * np.sign(phi_val))
                    hz_rows.append(0.0)
                    if i != len(phi_final) - 1:
                        t_rows.append(tau_us)
                        hx_rows.append(0.0)
                        hz_rows.append(Delta_0_mhz)

                pulse_df = pd.DataFrame({
                    "t (us)": t_rows,
                    "Omega_x (2pi MHz)": hx_rows,
                    "Omega_y (2pi MHz)": [0.0] * len(hx_rows),
                    "Omega_z (2pi MHz)": hz_rows,
                })

                plot_path = os.path.join(out_dir, f"u00_final_K={int(K)}.png")
                plot_matrix_element_vs_delta(
                    phi_final, cfg_sp, delta_vals, alpha_vals, out_path=plot_path
                )

                st.session_state["results"] = {
                    "final_loss": final_loss,
                    "pulse_df":   pulse_df,
                    "plot_path":  plot_path,
                    "phi_final":  phi_final.cpu(),
                    "delta_vals": delta_vals.cpu(),
                    "alpha_vals": alpha_vals.cpu(),
                    "cfg":        cfg_sp,
                    "omega_mhz":  Omega_max_mhz,
                }

        # Results
        if st.session_state["results"] is not None:
            res = st.session_state["results"]
            st.success(f"Training complete. Final loss: {res['final_loss']:.6e}")

            phi_final    = res["phi_final"]
            delta_vals_c = res["delta_vals"]
            alpha_vals_c = res["alpha_vals"]
            cfg_res      = res["cfg"]
            omega_mhz_r  = res["omega_mhz"]

            cfg_cpu = TrainConfig(
                Omega_max=cfg_res.Omega_max,
                Delta_0=cfg_res.Delta_0,
                robustness_window=cfg_res.robustness_window,
                K=cfg_res.K,
                device="cpu",
            )

            tab_pulse, tab_matrix, tab_fid, tab_phi = st.tabs([
                "Pulse Schedule", "Matrix element", "Fidelity", "φ vector",
            ])

            with tab_pulse:
                st.subheader("Pulse Schedule")
                st.dataframe(res["pulse_df"], use_container_width=True)
                st.download_button(
                    label="Download pulse schedule CSV",
                    data=res["pulse_df"].to_csv(index=False).encode("utf-8"),
                    file_name="pulse_schedule.csv",
                    mime="text/csv",
                )

            with tab_matrix:
                st.subheader("Matrix Element vs Detuning")
                if os.path.exists(res["plot_path"]):
                    st.image(res["plot_path"],
                             caption="Re(u₀₀) and Im(u₀₁) vs δ with target windows",
                             use_column_width=True)

            with tab_fid:
                with st.spinner("Evaluating fidelity …"):
                    fid     = fidelity(phi_final, delta_vals_c, alpha_vals_c, cfg_cpu)
                    runtime = get_control_runtime(phi_final, cfg_cpu)
                st.metric("Fidelity", f"{fid:.6f}")
                st.metric("Infidelity (1 − F)", f"{1 - fid:.2e}")
                st.metric("Runtime (μs)", f"{runtime:.4f}")

            with tab_phi:
                st.pyplot(_plot_phi_bar(phi_final.numpy()))
                plt.close("all")


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
