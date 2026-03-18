"""Scaling law study using the neural network pulse generator.

For each (Omega_max, K) pair:
  1. Train ONE model (or load existing weights)
  2. Run many trials with random alpha_vals via fast inference
  3. Evaluate fidelity and runtime

This is dramatically faster than the gradient-based scaling_law.py,
which trains from scratch for every single trial.

Usage:
    python -m neural_network_optimization.scaling_law --out_dir nn_scaling_law_results

    # Small test run:
    python -m neural_network_optimization.scaling_law --small True --out_dir nn_scaling_law_small
"""

import argparse
import itertools
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from neural_network_optimization.model import (
    phi_to_pulse_df,
    get_weight_path,
    DEFAULT_K,
    DEFAULT_OMEGA_MHZ,
    DELTA_CENTERS_MHZ,
    N_PEAKS,
)
from neural_network_optimization.inference import (
    load_model,
    generate_pulse,
    generate_phi,
    compute_fidelity,
    compute_runtime,
)
from neural_network_optimization.train import train

from single_pulse_optimization_QSP.qsp_fit_x_rotation import str_to_bool


def run_scaling_law(
    Omega_max_list,
    K_list,
    num_trials: int = 30,
    training_steps: int = 10000,
    out_dir: str = "nn_scaling_law_results",
):
    """Run the scaling law experiment with NN inference.

    For each (Omega_max, K) pair:
      1. Train model if no weights exist
      2. Run num_trials with random alpha_vals
      3. Record fidelity and runtime
    """
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    fidelity_data = {
        "Omega_max (MHz)": [],
        "K": [],
        "Runtime (us)": [],
        "trial": [],
        "Gate Fidelity": [],
    }

    total_configs = len(Omega_max_list) * len(K_list)
    total_trials = total_configs * num_trials

    print(f"Scaling law: {total_configs} model configs × {num_trials} trials = "
          f"{total_trials} evaluations")
    print(f"Omega_max: {Omega_max_list} MHz")
    print(f"K: {K_list}")
    print()

    start_t = time.time()

    for config_idx, (Omega_max, K) in enumerate(
        itertools.product(Omega_max_list, K_list)
    ):
        print(f"\n{'='*60}")
        print(f"Config {config_idx+1}/{total_configs}: "
              f"Omega_max={Omega_max} MHz, K={K}")
        print(f"{'='*60}")

        # Step 1: Train or load model
        weight_path = get_weight_path("neural_network_optimization", Omega_max, K)
        if os.path.exists(weight_path):
            print(f"  Loading existing weights from {weight_path}")
            model = load_model(Omega_mhz=Omega_max, K=K)
        else:
            print(f"  Training new model ({training_steps} steps)...")
            model = train(
                Omega_mhz=Omega_max,
                K=K,
                steps=training_steps,
                verbose=True,
            )
            model.eval()

        # Step 2: Run trials with random alpha_vals
        print(f"  Running {num_trials} trials...")
        trial_pbar = tqdm(range(num_trials), desc=f"  Trials Ω={Omega_max} K={K}")

        for trial in trial_pbar:
            # Random alpha_vals in [0, 2pi) for all peaks
            alpha_list = (2 * torch.rand(N_PEAKS)).tolist()  # in units of pi
            alpha_vals_rad = [a * math.pi for a in alpha_list]

            # Save input config
            config_tag = f"Omega_max{Omega_max}_K{K}_trial{trial + 1}"
            input_df = pd.DataFrame({
                "delta (MHz)": DELTA_CENTERS_MHZ,
                "alpha (pi rad)": alpha_list,
            })
            input_df.to_csv(
                os.path.join(data_dir, f"{config_tag}_input.csv"), index=False
            )

            # Inference
            fid = compute_fidelity(model, alpha_vals_rad)
            runtime = compute_runtime(model, alpha_vals_rad)

            # Save pulse CSV
            pulse_df = generate_pulse(model, alpha_vals_rad)
            pulse_df.to_csv(
                os.path.join(data_dir, f"{config_tag}.csv"), index=False
            )

            fidelity_data["Omega_max (MHz)"].append(Omega_max)
            fidelity_data["K"].append(K)
            fidelity_data["Runtime (us)"].append(runtime)
            fidelity_data["trial"].append(trial + 1)
            fidelity_data["Gate Fidelity"].append(fid)

            trial_pbar.set_postfix({"fid": f"{fid:.4f}", "rt_ns": f"{runtime*1e3:.1f}"})

        # Print summary for this config
        config_fids = [
            fidelity_data["Gate Fidelity"][i]
            for i in range(len(fidelity_data["Gate Fidelity"]))
            if (fidelity_data["Omega_max (MHz)"][i] == Omega_max
                and fidelity_data["K"][i] == K)
        ]
        print(f"  Mean fidelity: {np.mean(config_fids):.4f} "
              f"± {np.std(config_fids):.4f}")

    # Save results
    fidelity_df = pd.DataFrame(fidelity_data)
    csv_path = os.path.join(out_dir, "nn_scaling_law_fidelity.csv")
    fidelity_df.to_csv(csv_path, index=False)

    elapsed = time.time() - start_t
    print(f"\n{'='*60}")
    print(f"Scaling law complete in {elapsed:.1f}s")
    print(f"Results saved to {csv_path}")
    print(f"{'='*60}")

    return fidelity_df


def main():
    parser = argparse.ArgumentParser(
        description="Run NN-based scaling law experiments."
    )
    parser.add_argument("--out_dir", type=str, default="nn_scaling_law_results",
                        help="Output directory for results.")
    parser.add_argument("--small", type=str_to_bool, default=False,
                        help="Run a small test case.")
    parser.add_argument("--training_steps", type=int, default=10000,
                        help="Training steps per model (if not already trained).")
    parser.add_argument("--num_trials", type=int, default=30,
                        help="Number of random trials per config.")
    args = parser.parse_args()

    torch.manual_seed(42)

    Omega_max_list = [20, 40, 80]  # MHz
    K_list = [30, 50, 70, 100]
    num_trials = args.num_trials

    if args.small:
        Omega_max_list = [40, 80]
        K_list = [50, 70]
        num_trials = 4
        args.out_dir = "nn_scaling_law_small"

    run_scaling_law(
        Omega_max_list=Omega_max_list,
        K_list=K_list,
        num_trials=num_trials,
        training_steps=args.training_steps,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
