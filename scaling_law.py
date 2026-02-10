from qsp_fit_relaxed import *
import itertools
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
import pandas as pd
import argparse
import multiprocessing as mp
from datetime import datetime, timedelta
import time


def get_min_distance(Delta_0, sigma, delta_vals):
    distance = [abs((delta_vals[0] - sigma) + Delta_0), abs(Delta_0 - (delta_vals[-1] + sigma))]
    distance += [abs((delta_vals[i] - sigma) - (delta_vals[i-1] + sigma)) for i in range(1, len(delta_vals))]
    return min(distance)


def generate_delta_alpha_pairs(N, Delta_0, signal_window, random_delta=True):
    """
    Returns delta_list and alpha_list both of length N.

    delta are RANDOM value of detuning ranging in [-Delta_0, Delta_0]. The values of delta's must
    satisfy such that each of the intervals [delta - signal_window, delta + signal_window] are disjoint


    alpha_list are rotation angles in radians/pi unit. These are random values from [-1, -1]
    correspoinding to -pi rotation to +pi rotation.
    """
    if random_delta:
        delta_tensor = torch.linspace(-Delta_0 + signal_window, Delta_0 - signal_window, steps=N + 1)[:-1]
        delta_list = (delta_tensor + Delta_0 / (N + 1) + 0.3 * (Delta_0 / (N + 1) - signal_window) * (torch.rand(N) - 0.5)).tolist()

    else:
        delta_tensor = torch.linspace(-Delta_0 + Delta_0 / N, Delta_0 - Delta_0 / N, steps=N + 1)[:-1]

    alpha_list = [random.uniform(-1, 1) for _ in range(N)]

    return delta_list, alpha_list


def _run_single_trial(task):
    K, N, Delta_0, sigma_to_Delta_0, trial = task

    if sigma_to_Delta_0 is not None:
        signal_window = sigma_to_Delta_0 * Delta_0
    else:
        signal_window = Delta_0 / N
    Omega_max = 80  # Fix to 80 MHz

    cfg = TrainConfig(
        Omega_max=2 * math.pi * Omega_max,
        Delta_0=2 * math.pi * Delta_0,
        singal_window=2 * math.pi * signal_window,
        K=int(K),
        steps=3000,
        lr=0.05,
        device="cpu",
        end_with_W=False,
        out_dir="scaling_law_results/data",
        build_with_detuning=True,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)

    delta_list, alpha_list = generate_delta_alpha_pairs(N, Delta_0, signal_window, random_delta = sigma_to_Delta_0)
    delta_vals = torch.tensor(delta_list) * 2 * math.pi
    alpha_vals = torch.tensor(alpha_list) * math.pi

    config_tag = f"K{K}_N{N}_D{Delta_0}_S{sigma_to_Delta_0}_trial{trial + 1}"

    input_df = pd.DataFrame(
        {
            "delta (MHz)": delta_list,
            "alpha (pi rad)": alpha_list
        }
    )
    input_df.to_csv(os.path.join(cfg.out_dir, f"{config_tag}_input.csv"), index=False)

    phi_final, final_loss = train(
        cfg,
        delta_vals,
        alpha_vals,
        sample_size=2048,
        progress_cb=None,
        verbose=False
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
    runtime = get_control_runtime(phi_final, cfg)
    gate_fidelity = fidelity(phi_final, delta_vals, alpha_vals, cfg)

    return {
        "K": K,
        "N": N,
        "Delta_0 (MHz)": Delta_0,
        "sigma/Delta_0": sigma_to_Delta_0,
        "trial": trial + 1,
        "Runtime (us)": runtime,
        "min_distance (MHz)": get_min_distance(Delta_0, sigma_to_Delta_0 * Delta_0, delta_vals),
        "Gate Fidelity": gate_fidelity,
    }


def main():
    argparser = argparse.ArgumentParser(description="Run scaling law experiments.")
    argparser.add_argument("--out_dir", type=str, default="scaling_law_results", help="Output directory for results.")
    argparser.add_argument("--is_drive", type=str_to_bool, help="Whether to run on Google Drive.", default=False)
    argparser.add_argument("--small", type=str_to_bool, help="Whether to run a small test case.", default=False)
    args = argparser.parse_args()
    out_dir = "/content/drive/MyDrive/Colab Notebooks/Scaling Law/" if args.is_drive else args.out_dir

    K_list = [30, 50, 70, 100]
    N_list = [2, 3, 4]
    Delta_0_list = [50, 100, 150]
    # sigma_to_Delta_0_list = [0.02, 0.05, 0.1]
    sigma_to_Delta_0_list = [None]

    # # Small test case
    if args.small:
        K_list = [30, 50]
        N_list = [2, 4]
        Delta_0_list = [100]
        sigma_to_Delta_0_list = [0.05]

    num_trials = 12 if args.is_drive else 4


    fidelity_data = {
        "K": [],
        "N": [],
        "Delta_0 (MHz)": [],
        "sigma/Delta_0": [],
        "Runtime (us)": [],
        "min_distance (MHz)": [],
        "trial": [],
        "Gate Fidelity": [],
    }


    tasks = []
    for K, N, Delta_0, sigma_to_Delta_0 in itertools.product(
        K_list, N_list, Delta_0_list, sigma_to_Delta_0_list
    ):
        for trial in range(num_trials):
            tasks.append((K, N, Delta_0, sigma_to_Delta_0, trial))
    random.shuffle(tasks)

    # Keep worker count fixed (requested: 6) while respecting small task counts
    max_workers = min(os.cpu_count(), len(tasks)) if len(tasks) > 0 else 1
    print(f"Starting scaling law trials with {max_workers} parallel workers...")


    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_run_single_trial, task) for task in tasks]
        pbar = tqdm(total=len(futures), desc="Scaling-law trials", dynamic_ncols=True)
        start_t = time.time()
        total = len(futures)
        completed = 0

        def _fmt_hms(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        for fut in as_completed(futures):
            result = fut.result()
            completed += 1

            elapsed = time.time() - start_t
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total - completed) / rate if rate > 0 else float("inf")
            end_time = datetime.now() + timedelta(seconds=0 if remaining == float("inf") else remaining)

            pbar.update(1)
            if remaining != float("inf"):
                pbar.set_postfix_str(f"ETA {_fmt_hms(remaining)} | ends {end_time:%Y-%m-%d %H:%M:%S}")
                # print(f"ETA {_fmt_hms(remaining)} | ends {end_time:%Y-%m-%d %H:%M:%S}\n")
            else:
                pbar.set_postfix_str("ETA --:--:-- | ends --")

            fidelity_data["K"].append(result["K"])
            fidelity_data["N"].append(result["N"])
            fidelity_data["Delta_0 (MHz)"].append(result["Delta_0 (MHz)"])
            fidelity_data["sigma/Delta_0"].append(result["sigma/Delta_0"])
            fidelity_data["Runtime (us)"].append(result["Runtime (us)"])
            fidelity_data["min_distance (MHz)"].append(result["min_distance (MHz)"])
            fidelity_data["trial"].append(result["trial"])
            fidelity_data["Gate Fidelity"].append(result["Gate Fidelity"])

        pbar.close()

    fidelity_df = pd.DataFrame(fidelity_data)
    fidelity_df.to_csv(os.path.join(out_dir, "scaling_law_fidelity_results.csv"), index=False)


if __name__ == "__main__":
    main()
