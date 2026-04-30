from single_pulse_optimization_QSP.qsp_fit_x_rotation import *
from single_pulse_optimization_QSP.qsp_fit_x_rotation import LAMBDA_VAL

import itertools
import random
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import argparse
import math
import numpy as np

from neural_network_optimization_QSP import NNTrainConfig, QSPPhaseNet, train_nn, predict_phi


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


def _run_classical_batch(
    Omega_max: float,
    K: int,
    num_trials: int,
    out_dir: str,
    sample_size: int = 2048,
    steps: int = 8000,
    lr: float = 5e-2,
    device: str = "cpu",
) -> list:
    """
    Train `num_trials` independent QSP phase sequences in parallel on `device`.

    All trials share the same fixed delta_vals but get independent random alpha
    configs.  A single (num_trials, K+1) nn.Parameter is optimised with one Adam
    step per iteration; per-trial gradients are independent because the batch loss
    is the *sum* of per-trial losses.
    """
    Delta_0_mhz           = 200.0
    robustness_window_mhz = 10.0
    delta_list_mhz        = [-100.0, -32.0, 32.0, 100.0]
    N                     = 4
    B                     = num_trials

    Delta_0           = 2 * math.pi * Delta_0_mhz
    Omega             = 2 * math.pi * Omega_max
    robustness_window = 2 * math.pi * robustness_window_mhz

    dev = torch.device(device)

    delta_vals = torch.tensor(delta_list_mhz, dtype=torch.float64, device=dev) * 2 * math.pi  # (N,)

    # Each trial gets its own random alpha config in [0, 2π) rad
    alpha_vals_batch = 2 * math.pi * torch.rand(B, N, dtype=torch.float64, device=dev)  # (B, N)

    # Sample training points once (same convention as train() in qsp_fit_x_rotation.py).
    # Peaks are well-separated (min gap 64 MHz >> 2×sigma=20 MHz) so nearest-peak
    # lookup reduces to direct index lookup.
    idx          = torch.randint(0, N, (B, sample_size), device=dev)          # (B, S)
    centers      = delta_vals[idx]                                             # (B, S)
    jitter       = (2.0 * torch.rand(B, sample_size, dtype=torch.float64, device=dev) - 1.0) * robustness_window
    delta_sample = (centers + jitter).clamp(-Delta_0, Delta_0)                # (B, S)
    alpha_sample = alpha_vals_batch[torch.arange(B, device=dev).unsqueeze(1), idx]  # (B, S)

    # (B, K+1) batched parameter — one phase sequence per trial
    phi_batch = nn.Parameter(0.01 * torch.randn(B, K + 1, dtype=torch.float64, device=dev))
    opt   = torch.optim.Adam([phi_batch], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best_loss = torch.full((B,), float("inf"), dtype=torch.float64)
    best_phi  = torch.zeros(B, K + 1, dtype=torch.float64)

    target_u00 = (torch.cos(alpha_sample / 2) - 1j * torch.sin(alpha_sample / 2)).to(torch.complex128)

    with tqdm(total=steps, desc=f"  Omega={Omega_max} MHz  K={K}", dynamic_ncols=True, leave=False) as pbar:
        for _ in range(steps):
            U    = build_qsp_unitary_batched(phi_batch, delta_sample, Delta_0, Omega)  # (B, S, 2, 2)
            pred = U[:, :, 0, 0]                                                        # (B, S) complex

            fd_eps = 1e-2
            U_p        = build_qsp_unitary_batched(phi_batch, delta_sample + fd_eps, Delta_0, Omega)
            U_m        = build_qsp_unitary_batched(phi_batch, delta_sample - fd_eps, Delta_0, Omega)
            grad_pred  = (U_p[:, :, 0, 0] - U_m[:, :, 0, 0]) / (2 * fd_eps)

            err            = pred - target_u00
            loss_per_trial = (err.abs() ** 2 + LAMBDA_VAL * grad_pred.abs() ** 2).mean(dim=1)  # (B,)
            loss           = loss_per_trial.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([phi_batch], max_norm=10.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                improved = loss_per_trial.detach().cpu() < best_loss
                if improved.any():
                    best_loss[improved] = loss_per_trial.detach().cpu()[improved]
                    best_phi[improved]  = phi_batch.detach().cpu()[improved]

            pbar.set_postfix({"avg_loss": f"{loss_per_trial.mean().item():.3e}"})
            pbar.update(1)

    # Post-training: evaluate fidelity and save outputs per trial
    cfg_base = TrainConfig(
        Omega_max=Omega,
        Delta_0=Delta_0,
        robustness_window=robustness_window,
        K=K,
        device="cpu",
    )
    data_dir       = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    delta_vals_cpu = delta_vals.cpu()
    tau_us         = math.pi / (2.0 * Delta_0)

    results = []
    for b in range(B):
        phi     = best_phi[b]                      # (K+1,) on CPU
        alpha_b = alpha_vals_batch[b].cpu()        # (N,) in radians
        tag     = f"Omega_max{Omega_max}_K{K}_trial{b + 1}"

        pd.DataFrame({
            "delta (MHz)": delta_list_mhz,
            "alpha (pi rad)": (alpha_b / math.pi).tolist(),
        }).to_csv(os.path.join(data_dir, f"{tag}_input.csv"), index=False)

        t_rows, hx_rows, hz_rows = [], [], []
        for i, pv in enumerate(phi.tolist()):
            t_rows.append(abs(pv) / Omega)
            hx_rows.append(Omega_max * math.copysign(1.0, pv))
            hz_rows.append(0.0)
            if i != K:
                t_rows.append(tau_us)
                hx_rows.append(0.0)
                hz_rows.append(Delta_0_mhz)
        pd.DataFrame({
            "t (us)": t_rows,
            "H_x (2pi MHz)": hx_rows,
            "H_z (2pi MHz)": hz_rows,
        }).to_csv(os.path.join(data_dir, f"{tag}.csv"), index=False)

        fid     = fidelity(phi, delta_vals_cpu, alpha_b, cfg_base)
        runtime = get_control_runtime(phi, cfg_base)
        results.append({
            "Omega_max (MHz)": Omega_max,
            "K":               K,
            "trial":           b + 1,
            "Runtime (us)":    runtime,
            "Gate Fidelity":   fid,
        })

    return results



def _run_nn_scaling(task):
    """
    Train one QSPPhaseNet for a given (Omega_max, K) pair and evaluate it on a
    held-out test set of random α configurations (all peaks varying, peak_index=None).

    Returns a dict with Omega_max, K, train/eval loss statistics, and the
    average / std of gate fidelity across the test set.
    """
    Omega_max, K, out_dir, nn_steps, nn_batch_size, nn_sample_size, device = task

    Delta_0_mhz           = 200.0
    robustness_window_mhz = 10.0
    delta_vals_mhz        = [-100.0, -32.0, 32.0, 100.0]
    N                     = 4
    N_test                = 64   # held-out α configurations for fidelity eval

    run_dir = os.path.join(out_dir, "nn_data", f"Omega{Omega_max}_K{K}")
    os.makedirs(run_dir, exist_ok=True)

    cfg_nn = NNTrainConfig(
        K=K,
        N=N,
        Omega_max=2 * math.pi * Omega_max,
        Delta_0=2 * math.pi * Delta_0_mhz,
        robustness_window=2 * math.pi * robustness_window_mhz,
        delta_vals=[2 * math.pi * d for d in delta_vals_mhz],
        batch_size=nn_batch_size,
        steps=nn_steps,
        lr=1e-3,
        sample_size=nn_sample_size,
        peak_index=None,                      # all α vary
        device=device,
        out_dir=run_dir,
        eval_interval=max(100, nn_steps // 20),
        checkpoint_interval=nn_steps,         # single checkpoint at the end
        eval_configs=2048,
        lambda_grad=0.3,                      # gradient regularization to encourage smoothness (can help generalization
    )

    model_path = os.path.join(run_dir, "model_final.pt")
    if os.path.exists(model_path):
        print(f"  [cache hit] loading existing model from {model_path}")
        model = QSPPhaseNet(N=N, K=K).to(device).double()
        model.load_state_dict(torch.load(model_path, map_location=device))
        best_eval  = float("nan")
        final_eval = float("nan")
    else:
        model, _, eval_records = train_nn(cfg_nn, verbose=True)
        best_eval  = min(e for _, e in eval_records) if eval_records else float("nan")
        final_eval = eval_records[-1][1]            if eval_records else float("nan")

    # ── fidelity on held-out test set ─────────────────────────────────────────
    # phi, delta_vals_t, and alpha_i are all on CPU, so cfg_qsp must use "cpu"
    # regardless of the training device to avoid cross-device index errors.
    cfg_qsp = TrainConfig(
        Omega_max=cfg_nn.Omega_max,
        Delta_0=cfg_nn.Delta_0,
        robustness_window=cfg_nn.robustness_window,
        K=K,
        device="cpu",
    )
    delta_vals_t = torch.tensor(cfg_nn.delta_vals, dtype=torch.float64)

    torch.manual_seed(42)
    test_alphas = torch.rand(N_test, N, dtype=torch.float64) * 2 * math.pi

    fidelities, runtimes = [], []
    for alpha_i in test_alphas:
        phi = predict_phi(model, alpha_i, device=device).detach().cpu()
        fid = fidelity(phi, delta_vals_t.cpu(), alpha_i.cpu(), cfg_qsp)
        fidelities.append(fid)
        runtimes.append(get_control_runtime(phi, cfg_qsp))

    avg_fid = float(np.mean(fidelities))
    std_fid = float(np.std(fidelities))
    avg_runtime = float(np.mean(runtimes))
    std_runtime = float(np.std(runtimes))
    pi_pulse_us = math.pi / cfg_qsp.Omega_max          # pi / Omega  (µs)
    avg_runtime_per_pi_pulse = avg_runtime / pi_pulse_us

    # save per-config fidelity breakdown
    fid_df = pd.DataFrame({
        "alpha_0": test_alphas[:, 0].tolist(),
        "alpha_1": test_alphas[:, 1].tolist(),
        "alpha_2": test_alphas[:, 2].tolist(),
        "alpha_3": test_alphas[:, 3].tolist(),
        "gate_fidelity": fidelities,
        "runtime_us":    runtimes,
    })
    fid_df.to_csv(os.path.join(run_dir, "test_fidelities.csv"), index=False)

    print(
        f"  NN Omega={Omega_max} MHz  K={K}: "
        f"best_eval={best_eval:.4e}  avg_fidelity={avg_fid:.4f} ± {std_fid:.4f}  "
        f"avg_runtime={avg_runtime:.4f} µs  runtime/pi-pulse={avg_runtime_per_pi_pulse:.2f}"
    )

    return {
        "Omega_max (MHz)":          Omega_max,
        "K":                        K,
        "nn_steps":                 nn_steps,
        "final_eval_loss":          final_eval,
        "best_eval_loss":           best_eval,
        "avg_gate_fidelity":        avg_fid,
        "std_gate_fidelity":        std_fid,
        "avg_runtime_us":           avg_runtime,
        "std_runtime_us":           std_runtime,
        "avg_runtime_per_pi_pulse": avg_runtime_per_pi_pulse,
    }


"""
Number of peak and signal window is fixed for hBN sample. Specifically,
N = 4, signal_window (sigma) = (2pi) 10 MHz.

We fix the deltas to be (2pi) [-100, -32, 32, 100] MHz for the hBN sample.

Our objective is to study how infidelity and runtime varies with respect to:

1. Rabi frequency
2. K (QSP degree)
"""


def main():
    argparser = argparse.ArgumentParser(description="Run scaling law experiments.")
    argparser.add_argument("--out_dir", type=str, default="scaling_law_output")
    argparser.add_argument("--is_drive", type=str_to_bool, default=False,
                           help="Save to Google Drive path (Colab).")
    argparser.add_argument("--small", type=str_to_bool, default=False,
                           help="Run a quick smoke-test with reduced grids.")
    argparser.add_argument("--skip_classical", type=str_to_bool, default=False,
                           help="Skip the classical per-instance QSP trials.")
    argparser.add_argument("--skip_nn", type=str_to_bool, default=False,
                           help="Skip the neural-network scaling law.")
    argparser.add_argument("--nn_steps", type=int, default=10_000,
                           help="Training steps per NN model.")
    argparser.add_argument("--nn_batch_size", type=int, default=4096,
                           help="α configs per gradient step. 4096 saturates A100 80GB for all K.")
    argparser.add_argument("--nn_sample_size", type=int, default=512,
                           help="δ samples per α config per NN step.")
    argparser.add_argument("--num_trials", type=int, default=30,
                           help="Classical trials per (Omega, K) combination.")
    argparser.add_argument("--device", type=str, default="cpu",
                           help="Device for training (e.g. 'cpu' or 'cuda').")
    argparser.add_argument("--classical_steps", type=int, default=2000,
                           help="Gradient steps for classical per-instance optimization.")
    args = argparser.parse_args()

    out_dir = "/content/drive/MyDrive/Colab Notebooks/Scaling Law/" if args.is_drive else args.out_dir

    Omega_max_list = [160, 40, 80, 120]   # MHz
    K_list         = [100, 50, 70]
    num_trials     = args.num_trials
    nn_steps       = args.nn_steps
    nn_batch_size  = args.nn_batch_size
    nn_sample_size = args.nn_sample_size

    if args.small:
        Omega_max_list = [40, 80]
        K_list         = [50, 70]
        num_trials     = 2
        nn_steps       = 200
        nn_batch_size  = 16
        nn_sample_size = 64
        out_dir        = "scaling_law_small"

    os.makedirs(out_dir, exist_ok=True)

    # ── Classical per-instance QSP scaling law ────────────────────────────────

    if not args.skip_classical:
        fidelity_data = {
            "Omega_max (MHz)": [],
            "K":               [],
            "Runtime (us)":    [],
            "trial":           [],
            "Gate Fidelity":   [],
        }

        grid = list(itertools.product(Omega_max_list, K_list))
        print(f"\n=== Classical QSP scaling law (GPU-batched) ===")
        print(f"  Grid : Omega={Omega_max_list} MHz  K={K_list}")
        print(f"  Trials/config: {num_trials}  |  Configs: {len(grid)}  |  Device: {args.device}\n")

        for i, (Omega_max, K) in enumerate(grid, 1):
            print(f"[{i}/{len(grid)}] Omega={Omega_max} MHz  K={K}  ({num_trials} trials in parallel)")
            batch_results = _run_classical_batch(
                Omega_max=Omega_max,
                K=K,
                num_trials=num_trials,
                out_dir=out_dir,
                device=args.device,
                steps=args.classical_steps,
            )
            for r in batch_results:
                fidelity_data["Omega_max (MHz)"].append(r["Omega_max (MHz)"])
                fidelity_data["K"].append(r["K"])
                fidelity_data["Runtime (us)"].append(r["Runtime (us)"])
                fidelity_data["trial"].append(r["trial"])
                fidelity_data["Gate Fidelity"].append(r["Gate Fidelity"])

        fidelity_df = pd.DataFrame(fidelity_data)
        out_csv = os.path.join(out_dir, "scaling_law_classical.csv")
        fidelity_df.to_csv(out_csv, index=False)
        print(f"\nClassical results saved to {out_csv}")

    # ── Neural-network QSP scaling law ────────────────────────────────────────

    if not args.skip_nn:
        nn_grid = list(itertools.product(Omega_max_list, K_list))

        print(f"\n=== Neural-network QSP scaling law ===")
        print(f"  Grid : Omega={Omega_max_list} MHz  K={K_list}")
        print(f"  Steps/model: {nn_steps}  batch={nn_batch_size}  sample={nn_sample_size}")
        print(f"  Configs: {len(nn_grid)}  (run sequentially)\n")

        nn_results = []
        for idx, (Omega_max, K) in enumerate(nn_grid, 1):
            print(f"[{idx}/{len(nn_grid)}] Omega={Omega_max} MHz  K={K}")
            task = (Omega_max, K, out_dir, nn_steps, nn_batch_size, 
                    nn_sample_size, args.device)
            nn_results.append(_run_nn_scaling(task))

        nn_df = pd.DataFrame(nn_results)
        nn_csv = os.path.join(out_dir, "scaling_law_nn.csv")
        nn_df.to_csv(nn_csv, index=False)
        print(f"\nNN results saved to {nn_csv}")
        print(nn_df.to_string(index=False))


if __name__ == "__main__":
    main()
