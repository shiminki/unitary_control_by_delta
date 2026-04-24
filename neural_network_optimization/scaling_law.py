"""Scaling-law experiment over (Omega, K).

For each config:
  1. Load weights (train if missing).
  2. Sample N_EVAL_TRIALS alphas uniformly from [-EPS, 4*pi + EPS]^N_peaks.
  3. Record fidelity and runtime for each trial.
  4. Aggregate into summary.

Writes CSVs + three figures to outputs/ and PCA artefacts for one canonical config.
"""

import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from .constants import K_LIST, N_PEAKS, OMEGA_LIST
from .data import sample_alpha
from .inference import compute_fidelity, compute_runtime, load_model
from .model import get_weight_path
from .plotting import (
    plot_matrix_element,
    plot_runtime_vs_fidelity,
    plot_scaling_law,
)
from .train import train


def run_scaling_law(
    Omega_list=OMEGA_LIST,
    K_list=K_LIST,
    n_eval_trials: int = 512,
    epochs: int = 40,
    n_train: int = 65_536,
    n_eval: int = 1024,
    batch_size: int = 512,
    out_dir: str = "outputs",
    weight_dir: str = "neural_network_optimization/weights",
    device=None,
    verbose: bool = True,
    fidelity_sample_size: int = 2000,
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    t_start = time.time()

    for Omega, K in itertools.product(Omega_list, K_list):
        if verbose:
            print(f"\n=== Omega={Omega} MHz, K={K} ===")
        if os.path.exists(get_weight_path(weight_dir, Omega, K)):
            net = load_model(Omega=Omega, K=K, weight_dir=weight_dir, device=device)
        else:
            net = train(
                Omega=Omega, K=K, epochs=epochs,
                n_train=n_train, n_eval=n_eval, batch_size=batch_size,
                weight_dir=weight_dir, device=device, verbose=verbose,
                seed=seed,
            )
            net.eval()

        # Shared sampled alphas across trials
        alphas = sample_alpha(n_eval_trials, N_peaks=N_PEAKS).cpu().numpy()

        it = range(n_eval_trials)
        if verbose:
            it = tqdm(it, desc=f"Ω={Omega} K={K} eval")
        for trial in it:
            a = alphas[trial].tolist()
            fid = compute_fidelity(net, a, sample_size=fidelity_sample_size)
            rt = compute_runtime(net, a)
            row = {
                "Omega_mhz": float(Omega), "K": int(K),
                "trial": trial, "fidelity": float(fid), "runtime_us": float(rt),
            }
            for i, ai in enumerate(a):
                row[f"alpha_{i}"] = float(ai)
            rows.append(row)

    full_df = pd.DataFrame(rows)
    full_path = os.path.join(out_dir, "scaling_law_full.csv")
    full_df.to_csv(full_path, index=False)

    summary = (
        full_df.groupby(["Omega_mhz", "K"], sort=True)
        .agg(
            avg_fidelity=("fidelity", "mean"),
            min_fidelity=("fidelity", "min"),
            std_fidelity=("fidelity", "std"),
            mean_runtime_us=("runtime_us", "mean"),
            n_trials=("fidelity", "count"),
        )
        .reset_index()
    )
    summary_path = os.path.join(out_dir, "scaling_law_summary.csv")
    summary.to_csv(summary_path, index=False)

    plot_scaling_law(summary, os.path.join(out_dir, "scaling_law.png"))
    plot_runtime_vs_fidelity(full_df, os.path.join(out_dir, "runtime_vs_fidelity.png"))

    # Canonical matrix-element figure: pick (80, 70) if available, else first combo.
    (canon_omega, canon_K) = (80, 70) if (80 in Omega_list and 70 in K_list) else (Omega_list[0], K_list[0])
    net = load_model(Omega=canon_omega, K=canon_K, weight_dir=weight_dir, device=device)
    alpha_demo = [math.pi / 2, math.pi, math.pi / 3, 0.0][:N_PEAKS]
    plot_matrix_element(
        net, alpha_demo,
        os.path.join(out_dir, f"matrix_element_Omega{canon_omega}_K{canon_K}.png"),
    )

    # PCA for the canonical config across all peaks.
    from .pca_analysis import run_pca_all_peaks
    run_pca_all_peaks(
        net, out_dir=os.path.join(out_dir, "pca", f"Omega{canon_omega}_K{canon_K}"),
        n_samples=512,
    )

    if verbose:
        print(f"\nScaling law done in {time.time() - t_start:.1f}s.")
        print(f"  full:    {full_path}")
        print(f"  summary: {summary_path}")
        print(summary.to_string(index=False))

    return full_df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--weight_dir", type=str, default="neural_network_optimization/weights")
    ap.add_argument("--n_eval_trials", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--fidelity_sample_size", type=int, default=2000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--small", action="store_true",
                    help="Tiny grid for smoke testing.")
    args = ap.parse_args()

    if args.small:
        omega_list = [40]
        K_list = [8, 12]
        n_trials = 4
        epochs = 2
        n_train = 512
        n_eval = 128
        batch_size = 64
    else:
        omega_list = OMEGA_LIST
        K_list = K_LIST
        n_trials = args.n_eval_trials
        epochs = args.epochs
        n_train = 65_536
        n_eval = 1024
        batch_size = 512

    run_scaling_law(
        Omega_list=omega_list, K_list=K_list,
        n_eval_trials=n_trials, epochs=epochs,
        n_train=n_train, n_eval=n_eval, batch_size=batch_size,
        out_dir=args.out_dir, weight_dir=args.weight_dir,
        device=args.device, fidelity_sample_size=args.fidelity_sample_size,
    )


if __name__ == "__main__":
    main()
