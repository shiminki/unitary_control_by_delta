import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

import numpy as np
from numpy.polynomial import Chebyshev as T

import matplotlib.pyplot as plt
import os

import pandas as pd
from tqdm import tqdm

from qsp_fit import (Rz, W, bmm, DirectPhases, str_to_bool, build_U)



def build_U_with_detuning(
    phi: torch.Tensor,
    delta: torch.Tensor,
    Delta_0: float,
    Omega: float,
    end_with_W: bool = False,
    symmetric: bool = False,
) -> torch.Tensor:
    """
    Build symmetric unitary using only phi[0..K].

    Convention (default end_with_W=False):
      U = Rz(phi0) W Rz(phi1) W ... W Rz(phiK) W Rz(phi_{K-1}) ... W Rz(phi0)
    => starts with Rz and ends with Rz.

    If end_with_W=True:
      U = Rz(phi0) W Rz(phi1) W ... W Rz(phiK) W ... W Rz(phi0) W
    """
    assert phi.ndim == 1, "phi must be shape [K+1]"
    theta = (math.pi / 4) * (1 + delta / Delta_0)
    K = phi.shape[0] - 1
    B = theta.shape[0]
    dev = theta.device

    # Broadcast phases into batch
    # We'll build U by multiplying from left to right.
    U = torch.eye(2, dtype=torch.complex128, device=dev).expand(B, 2, 2).clone()
    Wt = W(theta)  # [B,2,2]

    # Helper to apply Rz of a scalar phase to the whole batch
    def apply_Rz_wth_detuning(Ucur: torch.Tensor, phase: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Goal is to apply R_z(phase) with control hamiltonian 
        H_c = Omega * sigma_z + delta * sigma_x
        for time t = phase / (2 * Omega)
        """
        phase_b = phase * (Omega / torch.sqrt(Omega ** 2 + delta ** 2))

        return bmm(Ucur, Rz(phase_b))

    # Forward half: Rz(phi0) then for j=0..K-1: W Rz(phi_{j+1})
    U = apply_Rz_wth_detuning(U, phi[0], delta)
    for j in range(0, K):
        U = bmm(U, Wt)
        U = apply_Rz_wth_detuning (U, phi[j + 1], delta)
    if symmetric:
        # Backward half: for j=K-1..0: W Rz(phi_j)
        for j in range(K - 1, -1, -1):
            U = bmm(U, Wt)
            U = apply_Rz_wth_detuning(U, phi[j], delta)

    if end_with_W:
        U = bmm(U, Wt)

    return U


# -------------------------
# Training loop
# -------------------------
@dataclass
class TrainConfig:
    Omega_max: float = 80 # Maximum Rabi frequency (MHz)
    Delta_0: float = 40  # Maximum detuning width (MHz); |delta| < Delta_0
    K: int = 30 # Number of QSP phases
    steps: int = 2000  # Number of training steps
    lr: float = 5e-3  # Learning rate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    end_with_W: bool = False
    print_every: int = 200
    plot_every: int = 500
    plot_points: int = 1024
    out_dir: str = "plots"
    combine_Q: bool = False  # Whether to combine Q operations into phases


def loss_fn(
    phi: torch.Tensor,
    theta_vals: torch.Tensor,
    alpha_vals: torch.Tensor,
    end_with_W: bool = False,
    lambda_val: float = 0.3
):
    """
    Docstring for loss_fn
    
    :param phi: phases of the QSP protocol
    :type phi: torch.Tensor
    :param theta_vals: Corresponding theta values of the given detunings.
        define theta = pi/4 * (1 + delta / Delta_0)
    :type theta_vals: torch.Tensor
    :param alpha_vals: Rotation angles, where when delta = delta_i, the resulting
        unitary should approximate Rz(alpha_i).
    :type alpha_vals: torch.Tensor
    :param end_with_W: Ends with signal operator for QSP protocol.
    :type end_with_W: bool
    :param lambda_val: weight for gradient term in loss
    :type lambda: float
    :return: loss value, predicted u_00 elements, target u_00 elements
    :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    # Ensure theta_vals is a leaf with grad tracking for d(pred)/d(theta)
    theta_vals = theta_vals.detach().clone().requires_grad_(True)
    U = build_U(phi, theta_vals, end_with_W=end_with_W)  # (N, 2, 2)
    pred = U[:, 0, 0]  # (N,) u_00 element
    target = torch.cos(alpha_vals / 2) + 1j * torch.sin(alpha_vals / 2)  # (N,)

    # Compute gradient: d(pred) / d(theta_vals)
    grad_pred = torch.autograd.grad(
        outputs=pred,
        inputs=theta_vals,
        grad_outputs=torch.ones_like(pred),  # elementwise derivative
        create_graph=True,  # if you want higher-order derivatives
        retain_graph=True   # if you need to reuse the graph
    )[0]

    target_err = pred - target
    
    loss = (target_err.abs() ** 2 + lambda_val * grad_pred.abs() ** 2).mean()
    return loss, pred, target


# Plot function

def expand_delta_alpha(
    delta_vals: torch.Tensor,
    alpha_vals: torch.Tensor,
    delta_range: float,
    samples_per_delta: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Expand each delta_i into samples within [delta_i - delta_range, delta_i + delta_range],
    and repeat the corresponding alpha_i for each sample.
    """
    if delta_vals.numel() != alpha_vals.numel():
        raise ValueError("delta_vals and alpha_vals must have the same length.")
    if samples_per_delta < 1:
        raise ValueError("samples_per_delta must be >= 1.")

    device = delta_vals.device
    dtype = delta_vals.dtype

    offsets = torch.linspace(
        -delta_range,
        delta_range,
        steps=samples_per_delta,
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    delta_expanded = delta_vals.unsqueeze(1) + offsets  # (N, M)
    alpha_expanded = alpha_vals.unsqueeze(1).expand(-1, samples_per_delta)  # (N, M)

    return delta_expanded.reshape(-1), alpha_expanded.reshape(-1)

def plot_u00_vs_delta(
    phi: torch.Tensor,
    Omega: float,
    delta_vals: torch.Tensor,
    alpha_vals: torch.Tensor,
    Delta_0: float,
    end_with_W: bool,
    device: torch.device,
    out_path: str,
    delta_width: float,
):
    """
    For delta = delta_vals[i] +/- delta_width/2, plot U00 vs delta and
    overlay the target U_target_00 within each segment.
    """
    
    delta_range = torch.linspace(-Delta_0, Delta_0, steps=1024, device=device)  # (N,)
    theta_range = (math.pi / 4) * (1 + delta_range / Delta_0)  # (N,)
    U = build_U(phi, theta_range.to(device), end_with_W=end_with_W)  # (N, 2, 2)
    u00 = U[:, 0, 0].detach().cpu()  # (N,)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(delta_range, u00.real, label="Re(u_00)")
    ax.plot(delta_range, u00.imag, label="Im(u_00)")


    theta_vals = (math.pi / 4) * (1 + delta_vals / Delta_0)  # (M,)

    for i, delta in enumerate(delta_vals):
        alpha_i = alpha_vals[i]
        delta_min = delta.item() - delta_width / 2
        delta_max = delta.item() + delta_width / 2

        ax.hlines(
            y=np.cos(alpha_i.item() / 2),
            xmin=delta_min,
            xmax=delta_max,
            colors="red",
            linestyles="dashed",
            label="Re(target u_00)" if i == 0 else None
        )
        ax.hlines(
            y=np.sin(alpha_i.item() / 2),
            xmin=delta_min,
            xmax=delta_max,
            colors="green",
            linestyles="dashed",
            label="Im(target u_00)" if i == 0 else None
        )

        ax.axvspan(delta_min, delta_max, color="gray", alpha=0.15)
        label = f"R_x({alpha_i / math.pi:.4f} pi)"
        ax.text(
            delta.item(),
            1.05,
            label,
            ha="center",
            va="bottom",
            fontsize=9
        )

    tau = math.pi/(8 * Delta_0)
    K = len(phi) - 1
    T = K * tau + sum(phi).item() / (2 * Omega)


    ax.set_xlabel("Detuning δ (MHz)")
    ax.set_ylabel("u_00 Element")
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f"QSP u_00 Element vs Target Controlled Rz(alpha)\nTotal Time T={T:.6f} μs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def main():
    torch.manual_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=30, help="Max phase index K (phi has length K+1).")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--num_peaks", type=int, default=4, help="Number of peaks in target function P(x).")
    ap.add_argument("--Delta_0", type=float, default=40.0, help="Maximum detuning width (MHz); |delta| < Delta_0")
    ap.add_argument("--Omega_max", type=float, default=80.0, help="Maximum Rabi frequency (MHz)")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--end_with_W", type=str_to_bool, help="If set, append a final W(theta) on the right.", default=False)
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--plot_every", type=int, default=5000)
    ap.add_argument("--plot_points", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default="plots_relaxed")
    args = ap.parse_args()


    os.makedirs(args.out_dir, exist_ok=True)

    cfg = TrainConfig(
        K=args.K,
        steps=args.steps,
        Delta_0=2 * math.pi * args.Delta_0,
        Omega_max=2 * math.pi * args.Omega_max,
        lr=args.lr,
        device=args.device,
        end_with_W=args.end_with_W,
        print_every=args.print_every,
        plot_every=args.plot_every,
        plot_points=args.plot_points,
        out_dir=args.out_dir,
    )

    print(cfg.end_with_W)

    device = torch.device(cfg.device)
    torch.set_default_dtype(torch.float64)

    phase_model = DirectPhases(cfg.K).to(device)

    K = args.K

    # define target detuning and angle delta_i, alpha_i

    delta_vals = torch.linspace(-cfg.Delta_0, cfg.Delta_0, steps=args.num_peaks, device=device)  # (N,)
    # alpha_vals = 2 * torch.rand(delta_vals.shape, device=device) * torch.pi  # (N,)

    alpha_vals = torch.tensor([
        0.0,
        math.pi / 2,
        math.pi,
        math.pi/3
    ], device=device)  # (N,)

    delta_vals_expand, alpha_vals_expand = expand_delta_alpha(
        delta_vals,
        alpha_vals,
        delta_range=cfg.Delta_0 / (4 * args.num_peaks),
        samples_per_delta=100
    )

    theta_vals_expand = (math.pi / 4) * (1 + delta_vals_expand / cfg.Delta_0)  # (N,)
        

    # Optimizer (plain gradient descent would be SGD; Adam is usually much easier)
    opt = torch.optim.Adam(phase_model.parameters(), lr=cfg.lr)

    # Optional: a mild scheduler can help
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    best_phi = None
    best_loss = float("inf")

    with tqdm(total=cfg.steps, desc="Training", dynamic_ncols=True) as pbar:
        for step in range(1, cfg.steps + 1):
            
            phi = phase_model()  # [K+1] real

            loss, pred, target = loss_fn(phi, theta_vals_expand, alpha_vals_expand, end_with_W=cfg.end_with_W)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_phi = phi.detach().cpu().clone()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phase_model.parameters(), max_norm=10.0)  # helps stability
            opt.step()
            sched.step()

            pbar.set_postfix({
                "step": step,
                "train_loss": f"{loss.item():.3e}"
            })
            
            pbar.update(1)

            if step % cfg.plot_every == 0 or step == cfg.steps:
                out_path = os.path.join(cfg.out_dir, f"u00_step{step:06d}_K={K}_combine_Q_{cfg.combine_Q}.png")
                plot_u00_vs_delta(phi, cfg.Omega_max, delta_vals, alpha_vals, cfg.Delta_0, cfg.end_with_W, cfg.device, out_path, delta_width=cfg.Delta_0 / (4*args.num_peaks))
                print(f"Saved plot: {out_path}")

    # Final phases
    phi_final = best_phi
    out_path = os.path.join(cfg.out_dir, f"u00_final_K={K}_combine_Q_{cfg.combine_Q}.png")
    plot_u00_vs_delta(phi_final, cfg.Omega_max, delta_vals, alpha_vals, cfg.Delta_0, cfg.end_with_W, cfg.device, out_path, delta_width=cfg.Delta_0 / (4*args.num_peaks))
    print(f"Saved plot: {out_path}")
    phi_df = pd.DataFrame({"index": np.arange(len(phi_final)), "phi": phi_final.numpy()})

    phi_df.to_csv(os.path.join(cfg.out_dir, f"learned_phases_K={K}_combine_Q_{cfg.combine_Q}.csv"), index=True)


if __name__ == "__main__":
    main()
