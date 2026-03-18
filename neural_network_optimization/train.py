"""Training pipeline for the joint neural network pulse generator.

Given Omega_max and QSP degree K, trains a JointPulseGeneratorNet that maps
alpha_vals (N rotation angles) -> QSP phases phi[0..K].

Default configuration:
    - 4 peaks at delta = (2pi) [-100, -32, 32, 100] MHz
    - Dataset: union of all-random and one-hot alpha_vals

Usage:
    # From project root:
    python -m neural_network_optimization.train --Omega_mhz 80 --K 70 --steps 10000

    # Train for a different Omega_max:
    python -m neural_network_optimization.train --Omega_mhz 40 --K 50 --steps 15000
"""

import argparse
import math
import os
import sys

import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from neural_network_optimization.model import (
    JointPulseGeneratorNet,
    presample_detunings_joint,
    compute_batch_loss_joint,
    sample_alpha_batch,
    get_weight_path,
    DEFAULT_K,
    DEFAULT_OMEGA_MHZ,
    N_PEAKS,
)


def train(
    Omega_mhz: float = DEFAULT_OMEGA_MHZ,
    K: int = DEFAULT_K,
    steps: int = 20000,
    lr: float = 5e-3,
    batch_size: int = 32,
    hidden_dim: int = 256,
    num_layers: int = 6,
    n_freq: int = 8,
    samples_per_peak: int = 128,
    resample_every: int = 0,
    device: str = "cpu",
    verbose: bool = True,
    out_dir: str = "neural_network_optimization",
    progress_cb=None,
) -> JointPulseGeneratorNet:
    """Train a JointPulseGeneratorNet and save weights.

    Parameters
    ----------
    Omega_mhz : Rabi frequency in MHz.
    K : QSP degree (phi has K+1 elements).
    steps : Number of training steps.
    lr : Learning rate.
    batch_size : Number of alpha_vals per training step.
    hidden_dim : Width of hidden layers.
    num_layers : Number of hidden layers.
    n_freq : Fourier input frequencies.
    samples_per_peak : Detuning samples per peak for loss computation.
    resample_every : Resample detunings every N steps (0 = never).
    device : "cpu" or "cuda".
    verbose : Print progress bar.
    out_dir : Directory for saving weights.
    progress_cb : Optional callback(step, total, loss, eta_seconds).

    Returns
    -------
    Trained JointPulseGeneratorNet.
    """
    torch.set_default_dtype(torch.float64)

    net = JointPulseGeneratorNet(
        K=K,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_freq=n_freq,
        Omega_mhz=Omega_mhz,
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    delta_all, peak_ids = presample_detunings_joint(
        net.delta_centers_ang,
        net.robustness_window_ang,
        net.Delta_0_ang,
        samples_per_peak,
        device,
    )

    best_loss = float("inf")
    best_state = None

    import time
    t0 = time.time()

    iterator = (
        tqdm(range(1, steps + 1), desc=f"Training Omega={Omega_mhz} K={K}")
        if verbose
        else range(1, steps + 1)
    )

    for step in iterator:
        if resample_every > 0 and step > 1 and step % resample_every == 0:
            delta_all, peak_ids = presample_detunings_joint(
                net.delta_centers_ang,
                net.robustness_window_ang,
                net.Delta_0_ang,
                samples_per_peak,
                device,
            )

        alpha_batch = sample_alpha_batch(batch_size, N_PEAKS, device)
        loss = compute_batch_loss_joint(net, alpha_batch, delta_all, peak_ids)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        sched.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.4e}", "best": f"{best_loss:.4e}"})

        if progress_cb is not None:
            elapsed = time.time() - t0
            rate = step / elapsed if elapsed > 0 else 0.0
            eta = (steps - step) / rate if rate > 0 else float("inf")
            progress_cb(step, steps, loss_val, eta)

    if best_state is not None:
        net.load_state_dict(best_state)

    # Save weights
    weight_path = get_weight_path(out_dir, Omega_mhz, K)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    torch.save(net.state_dict(), weight_path)

    if verbose:
        print(f"\nBest loss: {best_loss:.4e}")
        print(f"Weights saved to {weight_path}")

    return net


def main():
    parser = argparse.ArgumentParser(
        description="Train joint NN pulse generator for multi-peak QSP"
    )
    parser.add_argument("--Omega_mhz", type=float, default=DEFAULT_OMEGA_MHZ,
                        help="Rabi frequency in MHz")
    parser.add_argument("--K", type=int, default=DEFAULT_K,
                        help="QSP degree")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Training steps")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Alpha batch size per step")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="NN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="NN hidden layers")
    parser.add_argument("--n_freq", type=int, default=8,
                        help="Fourier input frequencies")
    parser.add_argument("--samples_per_peak", type=int, default=128,
                        help="Detuning samples per peak")
    parser.add_argument("--resample_every", type=int, default=0,
                        help="Resample detunings every N steps (0=never)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="neural_network_optimization",
                        help="Output directory for weights")
    args = parser.parse_args()

    torch.manual_seed(42)

    print("=" * 60)
    print(f"Training joint NN: Omega_max={args.Omega_mhz} MHz, K={args.K}")
    print(f"  steps={args.steps}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    print(f"  n_freq={args.n_freq}, samples_per_peak={args.samples_per_peak}")
    print("=" * 60)

    train(
        Omega_mhz=args.Omega_mhz,
        K=args.K,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_freq=args.n_freq,
        samples_per_peak=args.samples_per_peak,
        resample_every=args.resample_every,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
