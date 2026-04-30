#!/usr/bin/env python
"""
figures.py — Publication-quality figures for the paper.

Figure 1 : Matrix element vs detuning (2×2 grid, four target unitaries)
Figure 2 : Quantum-control scaling law (2×2 log-log, NN approach)
Figure 3 : Fidelity contour (4×1 grid, single shared colorbar)

Usage:
    python figures.py              # generate all figures
    python figures.py --fig 1     # only figure 1
    python figures.py --fig 2
    python figures.py --fig 3
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys

import matplotlib

from neural_network_optimization_QSP.qsp_phase_net import QSPPhaseNet
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm

import numpy as np
import pandas as pd
import torch

# ─── matplotlib style (academic paper) ───────────────────────────────────────

FS = 14                                 # base font size
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         FS,
    "axes.labelsize":    FS + 2,
    "axes.titlesize":    FS + 2,
    "xtick.labelsize":   FS,
    "ytick.labelsize":   FS,
    "legend.fontsize":   FS - 1,
    "lines.linewidth":   2.0,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
})

# ─── project imports ──────────────────────────────────────────────────────────

from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    TrainConfig,
    build_qsp_unitary,
    delta_to_theta,
    fidelity as qsp_fidelity,
    get_control_runtime,
    train,
)
from neural_network_optimization_QSP import (
    NNTrainConfig,
    train_nn,
    predict_phi,
)

# ─── paths and global parameters ─────────────────────────────────────────────

CACHE_DIR   = "phase_cache"        # single-pulse phase cache (shared with streamlit)
FIGURES_DIR = "figures"
NN_DIR      = "scaling_law_output/nn_data"   # NN model + result cache for scaling law

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(NN_DIR,      exist_ok=True)

# Standard hBN parameters (matching streamlit defaults)
DELTA_MHZ       = [-100.0, -32.0, 32.0, 100.0]   # peak detuning centres (MHz)
OMEGA_MAX_MHZ   = 80.0
DELTA_0_MHZ     = 200.0
ROBUSTNESS_MHZ  = 10.0
K_DEFAULT       = 70
STEPS_DEFAULT   = 8000

# Scaling-law grids
SL_OMEGA = [40, 80, 120, 160]   # MHz
SL_K     = [50, 70, 100]

# NN training budget for scaling law (increase for more accurate results)
NN_STEPS  = 5_000
NN_BATCH  = 64
NN_SAMPLE = 256

# Fidelity-contour grid resolution
N_DELTA = 100
N_EPS   = 40

# ─── phase-cache helpers ─────────────────────────────────────────────────────

def _cache_key(delta_mhz, alpha_pi, K, Delta_0=DELTA_0_MHZ,
               Omega=OMEGA_MAX_MHZ, sigma=ROBUSTNESS_MHZ):
    payload = json.dumps({
        "delta_vals":            [round(v, 8) for v in delta_mhz],
        "alpha_vals":            [round(v, 8) for v in alpha_pi],
        "K":                     int(K),
        "Delta_0_MHz":           round(Delta_0, 6),
        "Omega_max_MHz":         round(Omega,   6),
        "robustness_window_MHz": round(sigma,   6),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_phi(key):
    path = os.path.join(CACHE_DIR, f"phi_{key}.csv")
    if not os.path.exists(path):
        return None, None
    df  = pd.read_csv(path)
    phi = torch.tensor(df["phi"].values, dtype=torch.float64)
    loss = float(df["final_loss"].iloc[0]) if "final_loss" in df.columns else float("nan")
    return phi, loss


def _save_phi(key, phi, loss):
    path = os.path.join(CACHE_DIR, f"phi_{key}.csv")
    data = {
        "index":      list(range(len(phi))),
        "phi":        phi.cpu().numpy().tolist(),
        "final_loss": [float(loss)] + [float("nan")] * (len(phi) - 1),
    }
    pd.DataFrame(data).to_csv(path, index=False)


def get_phi(delta_mhz, alpha_pi,
            K=K_DEFAULT, steps=STEPS_DEFAULT,
            Omega=OMEGA_MAX_MHZ, Delta_0=DELTA_0_MHZ, sigma=ROBUSTNESS_MHZ):
    """Return (phi, TrainConfig).  Loads from cache; trains if missing."""
    key = _cache_key(delta_mhz, alpha_pi, K, Delta_0, Omega, sigma)
    phi, loss = _load_phi(key)
    cfg = TrainConfig(
        Omega_max=2 * math.pi * Omega,
        Delta_0=2 * math.pi * Delta_0,
        robustness_window=2 * math.pi * sigma,
        K=K,
        steps=steps,
        out_dir=FIGURES_DIR,
    )
    if phi is not None:
        print(f"[cache] key={key}  loss={loss:.3e}", flush=True)
        return phi, cfg

    print(f"[train] key={key}  alpha_pi={alpha_pi} ...", flush=True)
    delta_t = torch.tensor(delta_mhz, dtype=torch.float64) * (2 * math.pi)
    alpha_t = torch.tensor(alpha_pi,  dtype=torch.float64) * math.pi
    phi, final_loss, _ = train(
        cfg, delta_t, alpha_t, sample_size=2048, verbose=True,
        plot_name=os.path.join(FIGURES_DIR, f"_tmp_{key}.png"),
    )
    phi = phi.cpu()
    _save_phi(key, phi, final_loss)
    return phi, cfg

# ─── physics helpers ─────────────────────────────────────────────────────────

_H = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex128) / math.sqrt(2)

def _build_U_phys(phi, cfg, n_points=1024):
    """QSP unitary in physical basis over full Delta_0 range."""
    delta_range = torch.linspace(-cfg.Delta_0, cfg.Delta_0, n_points)
    U_qsp = build_qsp_unitary(phi, delta_range, cfg.Delta_0, cfg.Omega_max)
    U = _H @ U_qsp @ _H
    return delta_range, U


def _rx_target(alpha_i):
    """2×2 R_x(alpha_i) in physical basis."""
    c = math.cos(alpha_i / 2)
    s = math.sin(alpha_i / 2)
    U = torch.zeros((2, 2), dtype=torch.complex128)
    U[0, 0] = c;  U[0, 1] = -1j * s
    U[1, 0] = -1j * s;  U[1, 1] = c
    return U


def _gate_fidelity_batch(phi, delta_batch, alpha_i, cfg, omega_scale=1.0):
    """Gate fidelity (|Tr(U†_tgt U)|² + 2)/6 for each δ in batch."""
    delta_batch = delta_batch.cpu()
    U_qsp = build_qsp_unitary(phi.cpu(), delta_batch, cfg.Delta_0,
                               cfg.Omega_max * omega_scale)
    U = _H @ U_qsp @ _H
    Udagger = _rx_target(alpha_i).conj().T   # (2,2)
    traces = torch.einsum("ij,bji->b", Udagger, U)
    return (traces.abs() ** 2 + 2.0) / 6.0

# ─── NN scaling-law helpers ───────────────────────────────────────────────────

def run_nn_scaling(Omega_max_mhz, K):
    """Train / load a QSPPhaseNet for (Omega_max_mhz, K) and return result dict."""
    run_dir     = os.path.join(NN_DIR, f"Omega{int(Omega_max_mhz)}_K{K}")
    result_path = os.path.join(run_dir, "result.json")
    os.makedirs(run_dir, exist_ok=True)

    if os.path.exists(result_path):
        with open(result_path) as fh:
            cached = json.load(fh)
        print(f"[nn-cache] Omega={Omega_max_mhz}  K={K}  "
              f"infidelity={cached['avg_infidelity']:.4f}", flush=True)
        return cached

    print(f"[nn-train] Omega={Omega_max_mhz}  K={K} ...", flush=True)
    N             = 4
    Delta_0_mhz   = DELTA_0_MHZ
    sigma_mhz     = ROBUSTNESS_MHZ
    N_test        = 64

    cfg_nn = NNTrainConfig(
        K=K, N=N,
        Omega_max=2 * math.pi * Omega_max_mhz,
        Delta_0=2 * math.pi * Delta_0_mhz,
        robustness_window=2 * math.pi * sigma_mhz,
        delta_vals=[2 * math.pi * d for d in DELTA_MHZ],
        batch_size=NN_BATCH,
        steps=NN_STEPS,
        lr=1e-3,
        sample_size=NN_SAMPLE,
        peak_index=None,
        device="cpu",
        out_dir=run_dir,
        eval_interval=max(100, NN_STEPS // 20),
        checkpoint_interval=NN_STEPS,
        eval_configs=128,
    )
    if os.path.exists(os.path.join(run_dir, "model_final.pt")):
        model = QSPPhaseNet(N=N, K=K).to("cpu").double()
        model.load_state_dict(torch.load(os.path.join(run_dir, "model_final.pt"), map_location="cpu"))
    else:
        model, _, eval_records = train_nn(cfg_nn, verbose=True)

    cfg_qsp = TrainConfig(
        Omega_max=cfg_nn.Omega_max,
        Delta_0=cfg_nn.Delta_0,
        robustness_window=cfg_nn.robustness_window,
        K=K,
    )
    delta_t = torch.tensor(cfg_nn.delta_vals, dtype=torch.float64)

    torch.manual_seed(42)
    test_alphas = torch.rand(N_test, N, dtype=torch.float64) * 2 * math.pi

    fidelities, runtimes = [], []
    for alpha_i in test_alphas:
        phi = predict_phi(model, alpha_i, device="cpu").detach().cpu()
        fidelities.append(qsp_fidelity(phi, delta_t, alpha_i, cfg_qsp))
        runtimes.append(get_control_runtime(phi, cfg_qsp))

    result = {
        "Omega_max_mhz":  float(Omega_max_mhz),
        "K":              int(K),
        "avg_infidelity": float(1.0 - np.mean(fidelities)),
        "std_infidelity": float(np.std(fidelities)),
        "avg_runtime":    float(np.mean(runtimes)),
        "std_runtime":    float(np.std(runtimes)),
    }
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  → infidelity={result['avg_infidelity']:.4f}  "
          f"runtime={result['avg_runtime']:.4f} µs", flush=True)
    return result

# ─── Figure 1: matrix element plot ───────────────────────────────────────────

# Target unitaries: [U_0, U_1, U_2, U_3] where Ui = R_x(π) for index i, I elsewhere
TARGETS_FIG1 = [
    {"alpha_pi": [1.0, 0.0, 0.0, 0.0],
     "label":    r"$[R_x(\pi),\,I,\,I,\,I]$"},
    {"alpha_pi": [0.0, 1.0, 0.0, 0.0],
     "label":    r"$[I,\,R_x(\pi),\,I,\,I]$"},
    {"alpha_pi": [0.0, 0.0, 1.0, 0.0],
     "label":    r"$[I,\,I,\,R_x(\pi),\,I]$"},
    {"alpha_pi": [0.0, 0.0, 0.0, 1.0],
     "label":    r"$[I,\,I,\,I,\,R_x(\pi)]$"},
]
# Subplot panel letters
_PANEL = ["(a)", "(b)", "(c)", "(d)"]


def _plot_matrix_element_ax(ax, phi, cfg, delta_t, alpha_t, panel_letter, target_label):
    """Draw Re(u₀₀) and Im(u₀₁) vs detuning into ax."""
    delta_range, U = _build_U_phys(phi, cfg)

    delta_mhz = delta_range.numpy() / (2 * math.pi)
    u00 = U[:, 0, 0].detach().numpy()
    u01 = U[:, 0, 1].detach().numpy()

    l1, = ax.plot(delta_mhz, u00.real, color="C0", label=r"$\mathrm{Re}(u_{00})$")
    l2, = ax.plot(delta_mhz, u01.imag, color="C1", label=r"$\mathrm{Im}(u_{01})$")

    sigma_mhz = cfg.robustness_window / (2 * math.pi)
    for i, (delta_rad, alpha_rad) in enumerate(
            zip(delta_t.tolist(), alpha_t.tolist())):
        d_mhz = delta_rad / (2 * math.pi)
        ax.axvspan(d_mhz - sigma_mhz, d_mhz + sigma_mhz,
                   alpha=0.12, color="gray", zorder=0)
        target_re = math.cos(alpha_rad / 2)
        target_im = -math.sin(alpha_rad / 2)
        ax.hlines(target_re, d_mhz - sigma_mhz, d_mhz + sigma_mhz,
                  colors="C0", linestyles="--", linewidths=1.5,
                  label=(r"$\mathrm{Re}(\mathrm{target})$" if i == 0 else "_"))
        ax.hlines(target_im, d_mhz - sigma_mhz, d_mhz + sigma_mhz,
                  colors="C1", linestyles="--", linewidths=1.5,
                  label=(r"$\mathrm{Im}(\mathrm{target})$" if i == 0 else "_"))

    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(-DELTA_0_MHZ, DELTA_0_MHZ)

    # Panel letter + target label inside subplot
    ax.text(0.02, 0.97, panel_letter, transform=ax.transAxes,
            fontsize=FS + 1, fontweight="bold", va="top", ha="left")
    ax.text(0.98, 0.97, target_label, transform=ax.transAxes,
            fontsize=FS, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="none"))


def figure1():
    print("\n=== Figure 1: matrix element ===", flush=True)
    phi_list, cfg_list = [], []
    for t in TARGETS_FIG1:
        phi, cfg = get_phi(DELTA_MHZ, t["alpha_pi"])
        phi_list.append(phi)
        cfg_list.append(cfg)

    # Write stats to stdout and file
    stats_lines = ["Figure 1 stats — single-pulse QSP optimization\n" + "="*50]
    for i, (t, phi, cfg) in enumerate(zip(TARGETS_FIG1, phi_list, cfg_list)):
        delta_t = torch.tensor(DELTA_MHZ, dtype=torch.float64) * (2 * math.pi)
        alpha_t = torch.tensor(t["alpha_pi"], dtype=torch.float64) * math.pi
        fid = qsp_fidelity(phi, delta_t, alpha_t, cfg)
        T   = get_control_runtime(phi, cfg)
        line = (f"Subplot {_PANEL[i]}  {t['label']}:\n"
                f"  Total gate time  T = {T:.6f} µs\n"
                f"  Gate fidelity    F = {fid:.6f}")
        stats_lines.append(line)
        print(line, flush=True)

    stats_path = os.path.join(FIGURES_DIR, "figure1_stats.txt")
    with open(stats_path, "w") as fh:
        fh.write("\n\n".join(stats_lines) + "\n")
    print(f"Stats written to {stats_path}", flush=True)

    # 2×2 figure with shared axes
    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                              sharex=True, sharey=True,
                              gridspec_kw={"hspace": 0.06, "wspace": 0.06})

    for idx, (t, phi, cfg) in enumerate(zip(TARGETS_FIG1, phi_list, cfg_list)):
        ax = axes[idx // 2, idx % 2]
        delta_t = torch.tensor(DELTA_MHZ, dtype=torch.float64) * (2 * math.pi)
        alpha_t = torch.tensor(t["alpha_pi"], dtype=torch.float64) * math.pi
        _plot_matrix_element_ax(ax, phi, cfg, delta_t, alpha_t,
                                _PANEL[idx], t["label"])

    # Axis labels (only outer edges)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Detuning $\delta$ (MHz)", labelpad=4)
    for ax in axes[:, 0]:
        ax.set_ylabel("Matrix Element", labelpad=4)

    # Single legend on top-left subplot
    axes[0, 0].legend(loc="lower right", fontsize=FS - 1, framealpha=0.9)

    out = os.path.join(FIGURES_DIR, "figure1_matrix_element.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

# ─── Figure 2: scaling law ────────────────────────────────────────────────────

_OMEGA_COLORS = {40: "C0", 80: "C1", 120: "C2", 160: "C3"}
_K_COLORS     = {50: "C0", 70: "C1", 100: "C2"}
_MARKERS      = ["o", "s", "^", "D"]


def _fit_power_law_2d(df, y_col):
    """Fit y ~ C * Omega^n1 * K^n2 via OLS in log-log space. Returns (C, n1, n2, r2)."""
    mask = df[y_col] > 0
    sub  = df[mask]
    log_y     = np.log(sub[y_col].values)
    log_Omega = np.log(sub["Omega_max_mhz"].values)
    log_K     = np.log(sub["K"].values)
    X = np.column_stack([np.ones(len(sub)), log_Omega, log_K])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_y, rcond=None)
    log_C, n1, n2 = coeffs
    y_pred = X @ coeffs
    ss_res = np.sum((log_y - y_pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(log_C)), float(n1), float(n2), float(r2)


def _fit_power_law_1d(x_vals, y_vals):
    """Fit y ~ C * x^n via OLS in log space. Returns (C, n, r2)."""
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    mask = (y > 0) & (x > 0)
    lx, ly = np.log(x[mask]), np.log(y[mask])
    if len(lx) < 2:
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones(len(lx)), lx])
    coeffs, _, _, _ = np.linalg.lstsq(X, ly, rcond=None)
    log_C, n = coeffs
    ss_res = np.sum((ly - X @ coeffs) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(np.exp(log_C)), float(n), float(r2)


def figure2():
    print("\n=== Figure 2: scaling law ===", flush=True)

    rows = []
    for Omega, K in itertools.product(SL_OMEGA, SL_K):
        rows.append(run_nn_scaling(Omega, K))
    df = pd.DataFrame(rows)

    # ── scaling-law fits ──────────────────────────────────────────────────────
    lines = ["Figure 2 — Scaling Law Fits", "=" * 60, ""]

    for y_col, label in [("avg_infidelity", "Infidelity (1-F)"),
                          ("avg_runtime",    "Runtime T (µs)")]:
        C, n1, n2, r2 = _fit_power_law_2d(df, y_col)
        lines.append(f"{label}")
        lines.append(f"  2-D fit:  y ~ {C:.4e} * Omega^({n1:.4f}) * K^({n2:.4f})   R²={r2:.4f}")
        lines.append("")

        lines.append("  Varying Omega (fixed K):")
        for K_val in SL_K:
            sub = df[df["K"] == K_val].sort_values("Omega_max_mhz")
            C1, n, r2_1 = _fit_power_law_1d(sub["Omega_max_mhz"], sub[y_col])
            lines.append(f"    K={K_val:3d}:  y ~ {C1:.4e} * Omega^({n:.4f})   R²={r2_1:.4f}")

        lines.append("  Varying K (fixed Omega):")
        for Omega_val in SL_OMEGA:
            sub = df[df["Omega_max_mhz"] == Omega_val].sort_values("K")
            C1, n, r2_1 = _fit_power_law_1d(sub["K"], sub[y_col])
            lines.append(f"    Omega={Omega_val:3d} MHz:  y ~ {C1:.4e} * K^({n:.4f})   R²={r2_1:.4f}")
        lines.append("")

    lines.append("Raw data")
    lines.append("-" * 60)
    lines.append(df[["Omega_max_mhz", "K", "avg_infidelity", "std_infidelity",
                      "avg_runtime", "std_runtime"]].to_string(index=False))

    stats_path = os.path.join(FIGURES_DIR, "figure2_scaling_law_stats.txt")
    with open(stats_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Scaling-law stats written to {stats_path}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    def _loglog_panel(ax, x_col, y_col, std_col, group_col, group_vals, colors,
                      xlabel, ylabel, label_fmt, panel_letter):
        ax.set_xscale("log")
        ax.set_yscale("log")
        all_x = []
        for i, gv in enumerate(group_vals):
            sub  = df[df[group_col] == gv].sort_values(x_col)
            y    = sub[y_col].values
            x    = sub[x_col].values
            std  = sub[std_col].values
            mask = y > 0
            all_x.extend(x[mask].tolist())
            # clip lower error bar so it never crosses zero on log scale
            yerr_lo = np.minimum(std[mask], y[mask] * 0.999)
            ax.errorbar(x[mask], y[mask],
                        yerr=[yerr_lo, std[mask]],
                        marker=_MARKERS[i % len(_MARKERS)],
                        color=colors[gv],
                        label=label_fmt(gv),
                        markersize=7,
                        capsize=4,
                        elinewidth=1.2,
                        capthick=1.2)
        # tight x limits with ~20 % log-space padding on each side
        x_min, x_max = min(all_x), max(all_x)
        pad = (x_max / x_min) ** 0.2
        ax.set_xlim(x_min / pad, x_max * pad)
        ax.set_xlabel(xlabel, labelpad=4)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.legend(fontsize=FS - 2, loc="best")
        ax.yaxis.set_minor_locator(mticker.LogLocator(
            base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.grid(True, which="both", alpha=0.25)
        ax.text(0.04, 0.96, panel_letter, transform=ax.transAxes,
                fontsize=FS + 1, fontweight="bold", va="top", ha="left")

    # (a) infidelity vs Ω, grouped by K
    _loglog_panel(
        axes[0, 0],
        x_col="Omega_max_mhz", y_col="avg_infidelity", std_col="std_infidelity",
        group_col="K", group_vals=SL_K, colors=_K_COLORS,
        xlabel=r"$\Omega_{\mathrm{max}}$ (MHz)",
        ylabel=r"Infidelity $1-F$",
        label_fmt=lambda v: f"$K={v}$",
        panel_letter="(a)",
    )
    # (b) infidelity vs K, grouped by Ω
    _loglog_panel(
        axes[0, 1],
        x_col="K", y_col="avg_infidelity", std_col="std_infidelity",
        group_col="Omega_max_mhz", group_vals=SL_OMEGA, colors=_OMEGA_COLORS,
        xlabel=r"$K$",
        ylabel=r"",
        label_fmt=lambda v: rf"$\Omega_{{\mathrm{{max}}}}={int(v)}$ MHz",
        panel_letter="(b)",
    )
    # (c) runtime vs Ω, grouped by K
    _loglog_panel(
        axes[1, 0],
        x_col="Omega_max_mhz", y_col="avg_runtime", std_col="std_runtime",
        group_col="K", group_vals=SL_K, colors=_K_COLORS,
        xlabel=r"$\Omega_{\mathrm{max}}$ (MHz)",
        ylabel=r"Runtime $T$ ($\mu$s)",
        label_fmt=lambda v: f"$K={v}$",
        panel_letter="(c)",
    )
    # (d) runtime vs K, grouped by Ω
    _loglog_panel(
        axes[1, 1],
        x_col="K", y_col="avg_runtime", std_col="std_runtime",
        group_col="Omega_max_mhz", group_vals=SL_OMEGA, colors=_OMEGA_COLORS,
        xlabel=r"$K$",
        ylabel=r"",
        label_fmt=lambda v: rf"$\Omega_{{\mathrm{{max}}}}={int(v)}$ MHz",
        panel_letter="(d)",
    )

    out = os.path.join(FIGURES_DIR, "figure2_scaling_law.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

# ─── Figure 3: fidelity contour ───────────────────────────────────────────────

CONTOUR_LEVELS = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0]
_CMAP = "viridis"

_PEAK_LABELS = [
    r"$\delta_0 = -100$ MHz",
    r"$\delta_1 = -32$ MHz",
    r"$\delta_2 = +32$ MHz",
    r"$\delta_3 = +100$ MHz",
]


def _fidelity_contour_ax(ax, phi, cfg, delta_i, alpha_i):
    """
    Draw fidelity contour (detuning × calibration-error) into ax.
    Returns the contourf QuadContourSet for colorbar sharing.
    """
    sigma = cfg.robustness_window
    d_lo  = max(-cfg.Delta_0, delta_i - sigma)
    d_hi  = min(cfg.Delta_0,  delta_i + sigma)
    delta_range = torch.linspace(d_lo, d_hi, N_DELTA, dtype=torch.float64)
    eps_vals    = np.linspace(-0.1, 0.1, N_EPS)

    F_grid = np.zeros((N_DELTA, N_EPS))
    for j, eps in enumerate(eps_vals):
        F_grid[:, j] = _gate_fidelity_batch(
            phi, delta_range, alpha_i, cfg, 1.0 + eps).numpy()
        
    F_avg = F_grid.mean()
    print(f"  → avg fidelity over contour = {F_avg:.4f}", flush=True)

    delta_mhz   = delta_range.numpy() / (2 * math.pi)
    eps_pct     = eps_vals * 100
    delta_i_mhz = delta_i / (2 * math.pi)
    sigma_mhz   = sigma   / (2 * math.pi)

    cf = ax.contourf(delta_mhz, eps_pct, F_grid.T,
                     levels=CONTOUR_LEVELS, cmap=_CMAP, extend="min")
    ax.contour(delta_mhz, eps_pct, F_grid.T,
               levels=[0.95, 0.99, 0.999], colors="white", linewidths=0.8)
    ax.axvline(delta_i_mhz - sigma_mhz, color="gray", ls=":", lw=1.2)
    ax.axvline(delta_i_mhz + sigma_mhz, color="gray", ls=":", lw=1.2)
    ax.axhline(0, color="white", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel(r"Detuning $\delta$ (MHz)", labelpad=4)

    # Fewer x-ticks so labels don't crowd
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))

    return cf


def figure3():
    print("\n=== Figure 3: fidelity contour ===", flush=True)

    # Use the [Rx(π), I, I, I] target
    alpha_pi = [1.0, 0.0, 0.0, 0.0]
    phi, cfg = get_phi(DELTA_MHZ, alpha_pi)

    delta_t = torch.tensor(DELTA_MHZ, dtype=torch.float64) * (2 * math.pi)
    alpha_t = torch.tensor(alpha_pi,  dtype=torch.float64) * math.pi

    # 1 row × 4 cols; shared y-axis (calibration error identical everywhere)
    fig, axes = plt.subplots(
        1, 4, figsize=(18, 5), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    last_cf = None
    for i, (delta_i, alpha_i, plabel, panel) in enumerate(zip(
            delta_t.tolist(), alpha_t.tolist(),
            _PEAK_LABELS, ["(a)", "(b)", "(c)", "(d)"])):

        ax = axes[i]
        cf = _fidelity_contour_ax(ax, phi, cfg, delta_i, alpha_i)
        last_cf = cf

        # Panel letter
        ax.text(0.04, 0.97, panel, transform=ax.transAxes,
                fontsize=FS + 1, fontweight="bold", va="top", ha="left",
                color="white")
        # Peak label (inside, near top-center)
        ax.text(0.50, 0.97, plabel, transform=ax.transAxes,
                fontsize=FS - 1, va="top", ha="center",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black",
                          alpha=0.45, ec="none"))

    # y-axis label only on leftmost subplot (sharey handles tick labels)
    axes[0].set_ylabel(r"Calibration error $\varepsilon$ (%)", labelpad=4)

    # Single shared colorbar to the right of all subplots
    cbar = fig.colorbar(last_cf, ax=axes.tolist(),
                        label=r"Gate Fidelity $F$",
                        fraction=0.020, pad=0.02,
                        ticks=[lv for lv in CONTOUR_LEVELS if lv <= 1.0])
    cbar.ax.set_yticklabels(
        [f"{v:.3f}" if v == 0.999 else f"{v:.2f}"
         for v in CONTOUR_LEVELS if v <= 1.0],
        fontsize=FS - 1,
    )
    cbar.ax.tick_params(labelsize=FS - 1)
    cbar.ax.yaxis.label.set_fontsize(FS)

    out = os.path.join(FIGURES_DIR, "figure3_fidelity_contour.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}", flush=True)

# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate paper figures.")
    ap.add_argument("--fig", type=int, choices=[1, 2, 3],
                    help="Which figure to generate (default: all).")
    args = ap.parse_args()

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    if args.fig is None or args.fig == 1:
        figure1()
    if args.fig is None or args.fig == 2:
        figure2()
    if args.fig is None or args.fig == 3:
        figure3()

    print("\nDone.  Figures saved to:", FIGURES_DIR, flush=True)
