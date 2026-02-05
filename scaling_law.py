from qsp_fit_relaxed import *
import itertools
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import pandas as pd
import multiprocessing as mp
import argparse


def generate_delta_alpha_pairs(N, Delta_0, signal_window):
    """
    Returns delta_list and alpha_list both of length N.

    delta are RANDOM value of detuning ranging in [-Delta_0, Delta_0]. The values of delta's must
    satisfy such that each of the intervals [delta - signal_window, delta + signal_window] are disjoint


    alpha_list are rotation angles in radians/pi unit. These are random values from [-1, -1]
    correspoinding to -pi rotation to +pi rotation.
    """

    delta_list = []
    attempts = 0
    max_attempts = 1000

    while len(delta_list) < N and attempts < max_attempts:
        candidate = random.uniform(-Delta_0 + (N + 1) * signal_window, Delta_0 - (N + 1) * signal_window)
        overlap = False
        for d in delta_list:
            if abs(candidate - d) < 2 * signal_window:
                overlap = True
                break
        if not overlap:
            delta_list.append(candidate)
        attempts += 1

    if len(delta_list) < N:
        raise ValueError("Could not generate non-overlapping delta values after many attempts.")

    alpha_list = [random.uniform(-1, 1) for _ in range(N)]

    return delta_list, alpha_list


def _run_single_trial(task):
    K, N, Delta_0, sigma_to_Delta_0, trial, out_dir = task
    signal_window = sigma_to_Delta_0 * Delta_0
    Omega_max = 80  # Fix to 80 MHz

    cfg = TrainConfig(
        Omega_max=2 * math.pi * Omega_max,
        Delta_0=2 * math.pi * Delta_0,
        singal_window=2 * math.pi * signal_window,
        K=int(K),
        steps=2000,
        lr=7e-2,
        device="cpu",
        end_with_W=False,
        out_dir=out_dir,
        build_with_detuning=True,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    delta_list, alpha_list = generate_delta_alpha_pairs(N, Delta_0, signal_window)
    delta_vals = torch.tensor(delta_list) * 2 * math.pi
    alpha_vals = torch.tensor(alpha_list) * math.pi

    config_tag = f"K{K}_N{N}_D{Delta_0}_S{sigma_to_Delta_0}_trial{trial + 1}"

    input_df = pd.DataFrame(
        {
            "delta (MHz)": delta_list,
            "alpha (pi rad)": alpha_list,
            "signal_window (MHz)": [signal_window] * N,
        }
    )
    input_df.to_csv(os.path.join(cfg.out_dir, f"{config_tag}_input.csv"), index=False)

    phi_final, final_loss = train(
        cfg,
        delta_vals,
        alpha_vals,
        sample_size=1024,
        progress_cb=None,
        verbose=False,
        plot_name=os.path.join(cfg.out_dir, f"{config_tag}.png")
    )

    tau_us = (math.pi / (4.0 * cfg.Delta_0))
    omega_2pi_mhz = float(Omega_max)
    delta_2pi_mhz = float(Delta_0)
    t_rows = []
    hx_rows = []
    hz_rows = []
    for i, phi in enumerate(phi_final.tolist()):
        t_rows.append(np.abs(phi) / cfg.Omega_max)
        hx_rows.append(omega_2pi_mhz * np.sign(phi))
        hz_rows.append(0.0)
        if i != len(phi_final) - 1:
            t_rows.append(tau_us)
            hx_rows.append(0.0)
            hz_rows.append(delta_2pi_mhz)

    pulse_df = pd.DataFrame(
        {
            "t (us)": t_rows,
            "H_x (2pi MHz)": hx_rows,
            "H_z (2pi MHz)": hz_rows,
        }
    )
    pulse_df.to_csv(os.path.join(cfg.out_dir, f"{config_tag}.csv"), index=False)

    gate_fidelity = fidelity(phi_final, delta_vals, alpha_vals, cfg)

    print(f"K={K}, Delta_0: {Delta_0} MHz, Delta_vals {delta_list}, Alpha_vals {alpha_list}, signal window {signal_window} => Gate Fidelity: {gate_fidelity:.6f}")

    return {
        "K": K,
        "N": N,
        "Delta_0 (MHz)": Delta_0,
        "sigma/Delta_0": sigma_to_Delta_0,
        "trial": trial + 1,
        "Gate Fidelity": gate_fidelity,
    }


def main():

    argparser = argparse.ArgumentParser(description="Run scaling law experiments.")
    argparser.add_argument("--out_dir", type=str, default="scaling_law_results", help="Output directory for results.")
    argparser.add_argument("--is_drive", type=str_to_bool, help="Whether to run on Google Drive.", default=False)
    args = argparser.parse_args()
    out_dir = "/content/drive/MyDrive/Colab Notebooks/Scaling Law/" if args.is_drive else args.out_dir

    K_list = [30, 50, 70, 100]
    N_list = [2, 4, 6]
    Delta_0_list = [50, 100, 150]
    sigma_to_Delta_0_list = [0.02, 0.05, 0.1]
    num_trials = 12 if args.is_drive else 8


    fidelity_data = {
        "K": [],
        "N": [],
        "Delta_0 (MHz)": [],
        "sigma/Delta_0": [],
        "trial": [],
        "Gate Fidelity": [],
    }


    tasks = []
    for K, N, Delta_0, sigma_to_Delta_0 in itertools.product(
        K_list, N_list, Delta_0_list, sigma_to_Delta_0_list,
    ):
        for trial in range(num_trials):
            tasks.append((K, N, Delta_0, sigma_to_Delta_0, trial, out_dir))

    max_workers = max(1, min(os.cpu_count() or 1, len(tasks)))

    print(f"Starting scaling law trials with {max_workers} parallel workers...")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:

    # with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_single_trial, task) for task in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scaling-law trials"):
            result = fut.result()
            fidelity_data["K"].append(result["K"])
            fidelity_data["N"].append(result["N"])
            fidelity_data["Delta_0 (MHz)"].append(result["Delta_0 (MHz)"])
            fidelity_data["sigma/Delta_0"].append(result["sigma/Delta_0"])
            fidelity_data["trial"].append(result["trial"])
            fidelity_data["Gate Fidelity"].append(result["Gate Fidelity"])
    
    fidelity_df = pd.DataFrame(fidelity_data)
    fidelity_df.to_csv(os.path.join(out_dir,"scaling_law_fidelity_results.csv"), index=False)


if __name__ == "__main__":
    main()
