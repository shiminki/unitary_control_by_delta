#!/usr/bin/env python3
"""
Gradient-based optimization of symmetric QSP-like phase sequence.

Goal: learn phi[0..K] so that U_phi(theta)[0,0] matches P(cos(theta)) in MSE.

U_phi(theta) = Rz(phi0) W Rz(phi1) W ... W Rz(phiK) W ... W Rz(phi0)
(ends with Rz(phi0), i.e. no trailing W at the far right).

If you want the alternative convention that ENDS with W, see `build_U(..., end_with_W=True)`.
"""

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

__all__ = ["TrainConfig", "train", "fidelity", "plot_u00_vs_theta", "str_to_bool"]


# -------------------------
# Target function P(x)
# -------------------------
def example_P(x: torch.Tensor, alpha: float = math.pi) -> torch.Tensor:
    """
    Example target: a piecewise-constant function (complex-valued) of x in [-1,1].
    Replace this with YOUR P(x).

    Here:
      P(x) = 1                       if |x| < 1/sqrt(2)
             exp(i * alpha)          if |x| >= 1/sqrt(2)
    """
    thresh = 1.0 / math.sqrt(2.0)
    phase = torch.where(x.abs() < thresh, torch.zeros_like(x), torch.full_like(x, alpha))
    return torch.exp(1j * phase)  # complex


# -------------------------
# Matrix primitives
# -------------------------
def Rz(phi: torch.Tensor) -> torch.Tensor:
    """
    Rz(phi) = exp(-i sigma_z phi/2) = diag(e^{-i phi/2}, e^{+i phi/2})

    phi: shape [B] or scalar tensor (real)
    returns: shape [B,2,2] complex
    """
    # Ensure complex dtype downstream
    phi = phi.to(dtype=torch.float64)
    e0 = torch.exp(-0.5j * phi)
    e1 = torch.exp(+0.5j * phi)
    B = phi.shape[0] if phi.ndim > 0 else 1
    out = torch.zeros((B, 2, 2), dtype=torch.complex128, device=phi.device)
    out[:, 0, 0] = e0.reshape(-1)
    out[:, 1, 1] = e1.reshape(-1)
    return out


def W(theta: torch.Tensor) -> torch.Tensor:
    """
    W(theta) = [[cos(theta),  i sin(theta)],
                [-i sin(theta), cos(theta)]]

    theta: shape [B] (real)
    returns: shape [B,2,2] complex
    """
    theta = theta.to(dtype=torch.float64)
    c = torch.cos(theta)
    s = torch.sin(theta)
    out = torch.zeros((theta.shape[0], 2, 2), dtype=torch.complex128, device=theta.device)
    out[:, 0, 0] = c
    out[:, 1, 1] = c
    out[:, 0, 1] = -1j * s
    out[:, 1, 0] = -1j * s
    return out


def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batch matmul for [B,2,2] @ [B,2,2]."""
    return torch.bmm(A, B)


# -------------------------
# Building U_phi(theta)
# -------------------------
def build_U(
    phi: torch.Tensor,
    theta: torch.Tensor,
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
    K = phi.shape[0] - 1
    B = theta.shape[0]
    dev = theta.device

    # Broadcast phases into batch
    # We'll build U by multiplying from left to right.
    U = torch.eye(2, dtype=torch.complex128, device=dev).expand(B, 2, 2).clone()
    Wt = W(theta)  # [B,2,2]

    # Helper to apply Rz of a scalar phase to the whole batch
    def apply_Rz(Ucur: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        phase_b = phase.expand(B)  # [B]
        return bmm(Ucur, Rz(phase_b))

    # Forward half: Rz(phi0) then for j=0..K-1: W Rz(phi_{j+1})
    U = apply_Rz(U, phi[0])
    for j in range(0, K):
        U = bmm(U, Wt)
        U = apply_Rz(U, phi[j + 1])

    if symmetric:
        # Backward half: for j=K-1..0: W Rz(phi_j)
        for j in range(K - 1, -1, -1):
            U = bmm(U, Wt)
            U = apply_Rz(U, phi[j])

    if end_with_W:
        U = bmm(U, Wt)

    return U


# -------------------------
# Phase parametrizations (optional)
# -------------------------
class DirectPhases(nn.Module):
    """Learn phi[0..K] directly as free parameters (unconstrained)."""
    def __init__(self, K: int, init_scale: float = 0.01):
        super().__init__()
        # raw are real parameters; optimization is unconstrained
        raw = init_scale * torch.randn(K + 1, dtype=torch.float64)
        self.raw = nn.Parameter(raw)

    def forward(self) -> torch.Tensor:
        return self.raw


class BoundedPhases(nn.Module):
    """
    Learn unconstrained raw, but map to (-pi, pi) (or other range) for stability.
    """
    def __init__(self, K: int, bound: float = math.pi, init_scale: float = 0.01):
        super().__init__()
        raw = init_scale * torch.randn(K + 1, dtype=torch.float64)
        self.raw = nn.Parameter(raw)
        self.bound = float(bound)

    def forward(self) -> torch.Tensor:
        # maps to (-bound, bound)
        return self.bound * torch.tanh(self.raw)


class MLPPhases(nn.Module):
    """
    Optional: use a small MLP to generate phases from the index j (more "expressible" / smooth priors).
    Still produces phi[0..K], but with a neural parameterization.

    Good if K is large and you want fewer parameters than K+1.
    """
    def __init__(self, K: int, hidden: int = 128, depth: int = 3, bound: float = math.pi):
        super().__init__()
        self.K = K
        self.bound = float(bound)

        layers = []
        in_dim = 16  # positional features
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self) -> torch.Tensor:
        # Build positional encoding for indices j=0..K
        j = torch.arange(self.K + 1, dtype=torch.float64, device=self.net[0].weight.device)
        t = j / max(1, self.K)  # in [0,1]
        # Fourier features
        freqs = torch.tensor([1, 2, 4, 8, 16, 32, 64], dtype=torch.float64, device=t.device)
        ang = 2 * math.pi * t[:, None] * freqs[None, :]
        feats = torch.cat([t[:, None], torch.sin(ang), torch.cos(ang)], dim=1)  # [K+1, 1+7+7=15]
        # pad to 16 dims
        feats = torch.cat([feats, torch.ones((self.K + 1, 1), dtype=torch.float64, device=t.device)], dim=1)
        raw = self.net(feats).squeeze(-1)  # [K+1]
        return self.bound * torch.tanh(raw)


# -------------------------
# Training loop
# -------------------------
@dataclass
class TrainConfig:
    K: int = 30
    steps: int = 5000
    batch_theta: int = 2048
    lr: float = 5e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    end_with_W: bool = False
    use_grid: bool = False
    grid_size: int = 4096
    print_every: int = 200
    plot_every: int = 500
    plot_points: int = 1024
    out_dir: str = "plots"
    combine_Q: bool = False


def sample_theta(cfg: TrainConfig, step: int, device: torch.device) -> torch.Tensor:
    if cfg.use_grid:
        # Fixed grid in [-pi, pi]
        # (deterministic expectation approximation)
        theta = torch.linspace(-math.pi, math.pi, cfg.grid_size, dtype=torch.float64, device=device)
        # To keep compute consistent with "batch_theta", optionally subsample cyclically
        if cfg.grid_size > cfg.batch_theta:
            start = (step * cfg.batch_theta) % (cfg.grid_size - cfg.batch_theta + 1)
            theta = theta[start : start + cfg.batch_theta]
        return theta
    else:
        # Random Monte Carlo sample in [-pi, pi]
        return 2 *math.pi * torch.rand(cfg.batch_theta, dtype=torch.float64, device=device) - math.pi


def loss_fn(
    phi: torch.Tensor,
    theta: torch.Tensor,
    P_func: Callable[[torch.Tensor], torch.Tensor],
    end_with_W: bool,
    combine_Q:bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (loss, pred00, target)
    """
    U = build_U(phi, theta, end_with_W=end_with_W)  # [B,2,2] complex
    pred1 = U[:, 0, 0]  # [B] complex
    x = torch.cos(theta)  # [B] real in [-1,1]
    target1 = P_func(x)     # [B] complex

    pred2 = U[:, 0, 1]  # [B] complex
    # target2 = -1j * torch.sqrt(1.0 - target1.abs() ** 2) * x.sign()  # [B] complex principal sqrt
    target2 = torch.zeros_like(pred2)  # [B] complex

    if combine_Q:
        pred = torch.cat([pred1, pred2], dim=0)      # [2B]
        target = torch.cat([target1, target2], dim=0)  # [2B]
    else:
        pred = pred1
        target = target1  

    err = pred - target 

    loss = (err.abs() ** 2).mean()
    return loss, pred, target


# -------------------------
# Visualization
# -------------------------
@torch.no_grad()
def plot_u00_vs_theta(
    phi: torch.Tensor,
    P_func: Callable[[torch.Tensor], torch.Tensor],
    end_with_W: bool,
    device: torch.device,
    out_path: str,
    n_points: int = 1024,
    eps: float = 0.0,
):
    """
    2x2 plot:
      (0,0): U00 vs theta, overlay P(cos theta)
      (1,0): U00 vs x=cos(theta), overlay P(x)
      (0,1): U01 vs theta, overlay sqrt(1 - P(cos theta)^2)
      (1,1): U01 vs x=cos(theta), overlay sqrt(1 - P(x)^2)

    eps: optional tiny stabilizer inside sqrt: sqrt(1 - P^2 + eps)
         keep eps=0 unless you have numerical issues.
    """
    theta = torch.linspace(0.0, math.pi, n_points, dtype=torch.float64, device=device)
    x = torch.cos(theta)

    U = build_U(phi, theta, end_with_W=end_with_W)
    u00 = U[:, 0, 0]
    u01 = U[:, 0, 1]

    P = P_func(x)  # complex allowed
    Q = -1j * torch.sqrt(1.0 - P.abs() ** 2) * x.sign()  # complex principal sqrt

    # CPU numpy
    th = theta.detach().cpu().numpy()
    xx = x.detach().cpu().numpy()

    u00_np = u00.detach().cpu().numpy()
    u01_np = u01.detach().cpu().numpy()
    P_np = P.detach().cpu().numpy()
    Q_np = Q.detach().cpu().numpy()

    # Sort by x for x-plots
    order = xx.argsort()
    xx_s = xx[order]
    u00_s = u00_np[order]
    u01_s = u01_np[order]
    P_s = P_np[order]
    Q_s = Q_np[order]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(f"QSP output for deg={len(phi)-1} phases, duration {sum(abs(phi))/math.pi:.2f}pi", fontsize=16)

    # ---------- (0,0): U00 vs theta ----------
    ax = axs[0, 0]
    ax.plot(th, u00_np.real, label="Re(U00)")
    ax.plot(th, u00_np.imag, label="Im(U00)")
    ax.plot(th, abs(u00_np), label="|U00|", linestyle="--", alpha=0.8)
    ax.plot(th, P_np.real, label="Re(P(cosθ))", linewidth=2, alpha=0.75)
    ax.plot(th, P_np.imag, label="Im(P(cosθ))", linewidth=2, alpha=0.75)
    ax.set_title("U00 vs theta (target: P)")
    ax.set_xlabel("theta")
    ax.set_ylabel("value")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # ---------- (1,0): U00 vs x ----------
    ax = axs[1, 0]
    ax.plot(xx_s, u00_s.real, label="Re(U00)")
    ax.plot(xx_s, u00_s.imag, label="Im(U00)")
    ax.plot(xx_s, abs(u00_s), label="|U00|", linestyle="--", alpha=0.8)
    ax.plot(xx_s, P_s.real, label="Re(P(x))", linewidth=2, alpha=0.75)
    ax.plot(xx_s, P_s.imag, label="Im(P(x))", linewidth=2, alpha=0.75)
    ax.set_title("U00 vs x = cos(theta) (target: P)")
    ax.set_xlabel("x = cos(theta)")
    ax.set_ylabel("value")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # ---------- (0,1): U01 vs theta ----------
    ax = axs[0, 1]
    ax.plot(th, u01_np.real, label="Re(U01)")
    ax.plot(th, u01_np.imag, label="Im(U01)")
    ax.plot(th, abs(u01_np), label="|U01|", linestyle="--", alpha=0.8)
    ax.plot(th, Q_np.real, label="Re(Q(x)*sqrt(1-x^2))", linewidth=2, alpha=0.75)
    ax.plot(th, Q_np.imag, label="Im(Q(x)*sqrt(1-x^2))", linewidth=2, alpha=0.75)
    ax.set_title("U01 vs theta (target: sqrt(1 - P^2))")
    ax.set_xlabel("theta")
    ax.set_ylabel("value")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # ---------- (1,1): U01 vs x ----------
    ax = axs[1, 1]
    ax.plot(xx_s, u01_s.real, label="Re(U01)")
    ax.plot(xx_s, u01_s.imag, label="Im(U01)")
    ax.plot(xx_s, abs(u01_s), label="|U01|", linestyle="--", alpha=0.8)
    ax.plot(xx_s, Q_s.real, label="Re(sqrt(1-P^2))", linewidth=2, alpha=0.75)
    ax.plot(xx_s, Q_s.imag, label="Im(sqrt(1-P^2))", linewidth=2, alpha=0.75)
    ax.set_title("U01 vs x = cos(theta) (target: sqrt(1 - P^2))")
    ax.set_xlabel("x = cos(theta)")
    ax.set_ylabel("value")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def chebyshev_series_eval(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Evaluate sum_{n=0}^N c[n] * T_n(x) using Clenshaw recursion.
    c: (N+1,) complex or real
    x: (...,) real (typically in [-1, 1])
    returns: (...,) same dtype as c (complex if c is complex)
    """
    N = c.numel() - 1
    # b_{k+1}, b_{k+2}
    b1 = torch.zeros_like(x, dtype=c.dtype)
    b2 = torch.zeros_like(x, dtype=c.dtype)
    for j in range(N, 0, -1):
        b0 = 2 * x * b1 - b2 + c[j]
        b2, b1 = b1, b0
    return x * b1 - b2 + c[0]



def main():
    torch.manual_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=30, help="Max phase index K (phi has length K+1).")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_theta", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--end_with_W", type=str_to_bool, help="If set, append a final W(theta) on the right.")
    ap.add_argument("--use_grid", type=str_to_bool, help="Use a fixed theta grid instead of random sampling.")
    ap.add_argument("--grid_size", type=int, default=4096)
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--plot_every", type=int, default=500)
    ap.add_argument("--plot_points", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default="plots")
    ap.add_argument("--param", type=str, default="direct", choices=["direct", "bounded", "mlp"])
    ap.add_argument("--alpha", type=float, default=math.pi, help="Only used by the example P(x).")
    ap.add_argument("--combine_Q", type=str_to_bool, help="If set, combine U01 into loss computation.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = TrainConfig(
        K=args.K,
        steps=args.steps,
        batch_theta=args.batch_theta,
        lr=args.lr,
        device=args.device,
        end_with_W=args.end_with_W,
        use_grid=args.use_grid,
        grid_size=args.grid_size,
        print_every=args.print_every,
        plot_every=args.plot_every,
        plot_points=args.plot_points,
        out_dir=args.out_dir,
        combine_Q=args.combine_Q,
    )

    device = torch.device(cfg.device)
    torch.set_default_dtype(torch.float64)

    # Pick a phase parameterization
    if args.param == "direct":
        phase_model = DirectPhases(cfg.K).to(device)
    elif args.param == "bounded":
        phase_model = BoundedPhases(cfg.K, bound=math.pi).to(device)
    else:
        phase_model = MLPPhases(cfg.K, hidden=128, depth=3, bound=math.pi).to(device)

    # Define target P(x). Replace with your own callable if needed.
    alpha = torch.pi

    ansatz_coef = []

    K = args.K

    for k in range(K + 1):
        ansatz_coef.append(0.0) # idx 2k
        P_k = 4/torch.pi * (1 - (2*k + 1)/(2*K + 2)) * \
            (-1)**k / (2*k + 1) # idx 2k+1
        ansatz_coef.append(P_k)

    new_coef = []

    for i, c in enumerate(ansatz_coef):
        # new_coef.append(((1 - np.cos(alpha/2))/2 * c))
        new_coef.append(((1 - np.exp(1j*alpha/2))/2 * c))
        new_coef.append(0.0)

    new_coef[0] = (1 + np.exp(1j*alpha/2))/2

    new_coef = torch.tensor(new_coef, dtype=torch.complex128, device=device)

    def P_func(x: torch.Tensor) -> torch.Tensor:
        # x can be shape (...,), real
        x = x.to(device)
        return chebyshev_series_eval(new_coef, x)

    # Optimizer (plain gradient descent would be SGD; Adam is usually much easier)
    opt = torch.optim.Adam(phase_model.parameters(), lr=cfg.lr)

    # Optional: a mild scheduler can help
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    best_phi = None
    best_loss = float("inf")

    with tqdm(total=cfg.steps, desc="Training", dynamic_ncols=True) as pbar:
        for step in range(1, cfg.steps + 1):
            theta = sample_theta(cfg, step, device)
            phi = phase_model()  # [K+1] real

            loss, pred, target = loss_fn(phi, theta, P_func, end_with_W=cfg.end_with_W, combine_Q=cfg.combine_Q)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_phi = phi.detach().cpu().clone()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(phase_model.parameters(), max_norm=10.0)  # helps stability
            opt.step()
            sched.step()

            if step % cfg.print_every == 0 or step == 1:
                with torch.no_grad():
                    # Quick diagnostic on a small grid
                    th_eval = torch.linspace(0.0, math.pi, 512, device=device, dtype=torch.float64)
                    phi_eval = phase_model()
                    L_eval, _, _ = loss_fn(phi_eval, th_eval, P_func, end_with_W=cfg.end_with_W)

                    # Report a couple of samples
                    x_eval = torch.cos(th_eval)
                    U_eval = build_U(phi_eval, th_eval, end_with_W=cfg.end_with_W)
                    pred00 = U_eval[:, 0, 0]
                    tgt00 = P_func(x_eval)
                    max_err = (pred00 - tgt00).abs().max().item()

                pbar.set_postfix({
                    "step": step,
                    "train_loss": f"{loss.item():.3e}",
                    "eval_loss": f"{L_eval.item():.3e}",
                    "max_err": f"{max_err:.3e}",
                })
                print(f"[Step {step:06d}] train_loss={loss.item():.3e}, eval_loss={L_eval.item():.3e}, max_err={max_err:.3e}")
            
            pbar.set_postfix({
                "step": step,
                "train_loss": f"{loss.item():.3e}",
                "eval_loss": f"{L_eval.item():.3e}",
                "max_err": f"{max_err:.3e}",
            })
            
            pbar.update(1)

            if step % cfg.plot_every == 0 or step == cfg.steps:
                out_path = os.path.join(cfg.out_dir, f"u00_step{step:06d}_K={K}_combine_Q_{cfg.combine_Q}.png")
                plot_u00_vs_theta(
                    phi=phase_model(),
                    P_func=P_func,
                    end_with_W=cfg.end_with_W,
                    device=device,
                    out_path=out_path,
                    n_points=cfg.plot_points,
                )
                print(f"Saved plot: {out_path}")

    # Final phases
    phi_final = best_phi
    # print("\nLearned phases phi[0..K]:")
    # print(phi_final.numpy().tolist())
    phi_df = pd.DataFrame({"index": np.arange(len(phi_final)), "phi": phi_final.numpy()})

    phi_df.to_csv(os.path.join(cfg.out_dir, f"learned_phases_K={K}_combine_Q_{cfg.combine_Q}.csv"), index=True)


if __name__ == "__main__":
    main()
