"""Training loop for PulseNet."""

import argparse
import math
import os
import time

import torch
from tqdm import tqdm

from .constants import DEFAULT_K, OMEGA_MHZ
from .data import build_dataset
from .model import PulseNet, get_weight_path
from .physics import build_qsp_unitary_batched


def presample_detunings(
    delta_centers_ang,
    robustness_window_ang: float,
    Delta_0_ang: float,
    samples_per_peak: int = 128,
    device=None,
    generator=None,
):
    """Pre-sample detuning values within +/- robustness_window of each peak.

    Returns (delta_all, peak_ids) where delta_all is (N*S,) in angular units
    and peak_ids is (N*S,) long, indicating which peak each sample belongs to.
    """
    delta_list, peak_ids = [], []
    for j, center in enumerate(delta_centers_ang):
        jitter = (2.0 * torch.rand(samples_per_peak, dtype=torch.float64,
                                    device=device, generator=generator) - 1.0) * robustness_window_ang
        delta_s = (center + jitter).clamp(-Delta_0_ang, Delta_0_ang)
        delta_list.append(delta_s)
        peak_ids.append(torch.full((samples_per_peak,), j, dtype=torch.long, device=device))
    return torch.cat(delta_list), torch.cat(peak_ids)


def _compute_loss(
    net: "PulseNet",
    alpha_batch: torch.Tensor,
    delta_all: torch.Tensor,
    peak_ids: torch.Tensor,
) -> torch.Tensor:
    """Vectorized u_00 MSE loss using build_qsp_unitary_batched.

    For each sample (b, s) with detuning delta_all[s] near peak peak_ids[s]
    the target is u_00 = exp(-i alpha_batch[b, peak_ids[s]] / 2).
    """
    phi_batch = net(alpha_batch)  # (B, K+1)
    U_re, U_im = build_qsp_unitary_batched(
        phi_batch, delta_all, net.Delta_0_ang, net.Omega_ang,
    )  # each (B, D, 2, 2)
    pred_re = U_re[..., 0, 0]  # (B, D)
    pred_im = U_im[..., 0, 0]

    # alpha_targets[b, s] = alpha_batch[b, peak_ids[s]]
    alpha_targets = alpha_batch.index_select(1, peak_ids)  # (B, D)
    tgt_re = torch.cos(alpha_targets / 2)
    tgt_im = -torch.sin(alpha_targets / 2)

    err = (pred_re - tgt_re) ** 2 + (pred_im - tgt_im) ** 2
    return err.mean()


def train(
    Omega: float,
    K: int,
    *,
    n_train: int = 65_536,
    n_eval: int = 1024,
    batch_size: int = 512,
    epochs: int = 40,
    lr: float = 5e-3,
    samples_per_peak: int = 128,
    weight_dir: str = "neural_network_optimization/weights",
    device=None,
    seed: int = 0,
    verbose: bool = True,
) -> "PulseNet":
    """Train a PulseNet for the given (Omega, K).  Saves best-eval checkpoint.

    Returns the trained network (already loaded with best eval-loss weights).
    """
    torch.manual_seed(int(seed))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    net = PulseNet(Omega=Omega, K=K).to(device)

    alpha_train, alpha_eval = build_dataset(
        n_train=n_train, n_eval=n_eval, N_peaks=net.N_peaks, seed=seed, device=device,
    )

    delta_all, peak_ids = presample_detunings(
        net.delta_centers_ang, net.robustness_window_ang, net.Delta_0_ang,
        samples_per_peak=samples_per_peak, device=device,
    )

    steps_per_epoch = max(1, n_train // batch_size)
    total_steps = epochs * steps_per_epoch

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_state = None
    best_eval = float("inf")

    t0 = time.time()
    it = range(1, epochs + 1)
    if verbose:
        it = tqdm(it, desc=f"Omega={Omega} K={K}")

    for epoch in it:
        net.train()
        perm = torch.randperm(n_train, device=device)
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size : (step + 1) * batch_size]
            alpha_batch = alpha_train[idx]
            loss = _compute_loss(net, alpha_batch, delta_all, peak_ids)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            sched.step()

        # Eval
        net.eval()
        with torch.no_grad():
            eval_loss = 0.0
            n_chunks = max(1, (n_eval + batch_size - 1) // batch_size)
            for j in range(n_chunks):
                ab = alpha_eval[j * batch_size : (j + 1) * batch_size]
                if ab.shape[0] == 0:
                    continue
                eval_loss += _compute_loss(net, ab, delta_all, peak_ids).item() * ab.shape[0]
            eval_loss /= max(1, n_eval)

        if eval_loss < best_eval:
            best_eval = eval_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        if verbose and hasattr(it, "set_postfix"):
            it.set_postfix({"eval": f"{eval_loss:.3e}", "best": f"{best_eval:.3e}"})

    if best_state is not None:
        net.load_state_dict(best_state)

    os.makedirs(weight_dir, exist_ok=True)
    path = get_weight_path(weight_dir, Omega, K)
    torch.save(net.state_dict(), path)
    if verbose:
        print(f"[train] Omega={Omega} K={K} best_eval={best_eval:.3e} "
              f"time={time.time()-t0:.1f}s -> {path}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Omega", type=float, default=OMEGA_MHZ)
    ap.add_argument("--K", type=int, default=DEFAULT_K)
    ap.add_argument("--n_train", type=int, default=65_536)
    ap.add_argument("--n_eval", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_dir", type=str, default="neural_network_optimization/weights")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(
        Omega=args.Omega, K=args.K,
        n_train=args.n_train, n_eval=args.n_eval,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        weight_dir=args.weight_dir, device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
