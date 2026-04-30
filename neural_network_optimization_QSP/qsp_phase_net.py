"""
Neural-network amortised inference of QSP phase sequences.

Given N target gate encodings
    x = [cos(α₀/2), sin(α₀/2),  cos(α₁/2), sin(α₁/2),  …,  cos(α_{N-1}/2), sin(α_{N-1}/2)]
the model predicts a phase vector φ ∈ ℝ^{K+1} such that
    build_qsp_unitary(φ, δ) ≈ R_z(αᵢ)  for all δ ∈ [δᵢ − σ, δᵢ + σ], i = 0 … N-1.

Training is end-to-end differentiable: the QSP unitary is built from the
predicted φ and compared against the target R_z rotation, so no per-instance
classical solver is needed.

The fixed δ-peak locations are specified in NNTrainConfig.delta_vals and do
NOT change during training; the model generalises over α configurations.

CLI usage
---------
    python -m neural_network_optimization_QSP.qsp_phase_net \
        --K 70 --N 4 \
        --Omega_max 80 --Delta_0 200 --robustness_window 10 \
        --delta_vals -100 -32 32 100 \
        --steps 10000 --batch_size 64 --lr 1e-3

Programmatic usage
------------------
    from neural_network_optimization_QSP import NNTrainConfig, train_nn, predict_phi
    import math, torch

    cfg = NNTrainConfig(
        K=70, N=4,
        Omega_max=2*math.pi*80,
        Delta_0=2*math.pi*200,
        robustness_window=2*math.pi*10,
        delta_vals=[2*math.pi*d for d in [-100, -32, 32, 100]],
    )
    model, train_losses, eval_records = train_nn(cfg)

    alpha_query = torch.tensor([0.5*math.pi, math.pi, 0.3*math.pi, 1.5*math.pi])
    phi = predict_phi(model, alpha_query)   # shape (K+1,)
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from single_pulse_optimization_QSP.qsp_fit_x_rotation import (
    TrainConfig,
    build_qsp_unitary_batched,
    fidelity,
    plot_matrix_element_vs_delta,
)

__all__ = [
    "QSPPhaseNet",
    "NNTrainConfig",
    "train_nn",
    "predict_phi",
    "evaluate_fidelity",
    "batch_qsp_loss",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

class QSPPhaseNet(nn.Module):
    """
    MLP that maps target rotation encodings to a QSP phase vector.

    Input  : (…, 2N)   [cos(α₀/2), sin(α₀/2), …, cos(α_{N-1}/2), sin(α_{N-1}/2)]
    Output : (…, K+1)  φ = [φ₀, φ₁, …, φ_K]

    The cos/sin encoding preserves the periodicity of rotation angles and gives
    the network a smooth, bounded representation of the target gates.
    """

    def __init__(
        self,
        N: int,
        K: int,
        hidden_dims: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        N           : number of detuning peaks / target gates
        K           : QSP order; the output has K+1 phases
        hidden_dims : hidden-layer widths; defaults to a depth-4 MLP scaled to K
        """
        super().__init__()
        self.N = N
        self.K = K

        in_dim  = 2 * N
        out_dim = K + 1

        if hidden_dims is None:
            w = max(256, 4 * out_dim)
            hidden_dims = [w, w, w, w]

        layers: list = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (…, 2N) → φ : (…, K+1)"""
        return self.net(x)

    @staticmethod
    def encode_alphas(alpha: torch.Tensor) -> torch.Tensor:
        """
        Encode rotation angles as interleaved [cos(α/2), sin(α/2)] pairs.

        The encoding is interleaved per gate:
            [cos(α₀/2), sin(α₀/2), cos(α₁/2), sin(α₁/2), …]

        Parameters
        ----------
        alpha : (B, N) or (N,)  rotation angles in radians

        Returns
        -------
        (B, 2N) or (2N,)  encoded features in [-1, 1]
        """
        squeeze = alpha.ndim == 1
        if squeeze:
            alpha = alpha.unsqueeze(0)
        B, N = alpha.shape
        cos_h = torch.cos(alpha / 2)            # (B, N)
        sin_h = torch.sin(alpha / 2)            # (B, N)
        enc = torch.stack([cos_h, sin_h], dim=2).reshape(B, 2 * N)   # interleave
        return enc.squeeze(0) if squeeze else enc


# ─────────────────────────────────────────────────────────────────────────────
#  Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NNTrainConfig:
    """
    Configuration for neural-network QSP batch training.

    All frequency / detuning fields are in **angular units** (rad/μs).
    Convert from nominal MHz via:  value_rad = 2π × value_mhz.
    """

    # Problem geometry (required)
    K:                 int          # QSP order; φ has length K+1
    N:                 int          # Number of detuning peaks / target gates
    Omega_max:         float        # Maximum Rabi frequency (rad/μs)
    Delta_0:           float        # Max detuning range (rad/μs)
    robustness_window: float        # Half-width of robustness window (rad/μs)
    delta_vals:        List[float]  # Fixed peak detunings, length N (rad/μs)

    # α sampling range during training
    alpha_lo: float = 0.0
    alpha_hi: float = 2.0 * math.pi

    # Network architecture
    hidden_dims: Optional[List[int]] = None   # None → auto [w, w, w, w]

    # Optimisation
    batch_size:   int   = 64      # α configurations per gradient step
    steps:        int   = 10_000
    lr:           float = 1e-3
    sample_size:  int   = 256     # δ samples per α config per step
    lambda_grad:  float = 0.0     # finite-difference gradient-penalty weight

    # Single-peak mode
    peak_index: Optional[int] = None
    # If set to integer i, only α_i is drawn from Uniform(alpha_lo, alpha_hi)
    # during training and evaluation; all other α_j are fixed to 0.

    # Bookkeeping
    device:              str = "cpu"
    out_dir:             str = "nn_qsp_output"
    checkpoint_interval: int = 1000
    eval_interval:       int = 500
    eval_configs:        int = 128   # held-out α configs used for evaluation


# ─────────────────────────────────────────────────────────────────────────────
#  Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_training_data(
    alpha_configs:     torch.Tensor,   # (B, N)  sampled rotation angles
    delta_centers:     torch.Tensor,   # (N,)    fixed peak detunings
    robustness_window: float,
    Delta_0:           float,
    sample_size:       int,
    device:            str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each α configuration, sample δ values uniformly around every peak.

    Returns
    -------
    delta_s : (B, S)  detuning samples in rad/μs
    alpha_t : (B, S)  corresponding target rotation angles
    where S = (sample_size // N) * N.
    """
    B, N = alpha_configs.shape
    S_pp = max(1, sample_size // N)   # samples per peak

    d_parts, a_parts = [], []
    for i in range(N):
        jitter = (2.0 * torch.rand(B, S_pp, device=device) - 1.0) * robustness_window
        d_i = (delta_centers[i] + jitter).clamp(-Delta_0, Delta_0)
        a_i = alpha_configs[:, i : i + 1].expand(B, S_pp)
        d_parts.append(d_i)
        a_parts.append(a_i)

    return torch.cat(d_parts, dim=1), torch.cat(a_parts, dim=1)


def _sample_alpha_configs(
    B:          int,
    N:          int,
    alpha_lo:   float,
    alpha_hi:   float,
    device:     str,
    peak_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample a batch of α configurations of shape (B, N).

    peak_index=None : all N alphas drawn i.i.d. from Uniform(alpha_lo, alpha_hi).
    peak_index=i    : alpha_i ~ Uniform(alpha_lo, alpha_hi), alpha_j = 0 for j ≠ i.
    """
    if peak_index is None:
        return (
            torch.rand(B, N, device=device) * (alpha_hi - alpha_lo) + alpha_lo
        ).to(torch.float64)

    alphas = torch.zeros(B, N, dtype=torch.float64, device=device)
    alphas[:, peak_index] = (
        torch.rand(B, device=device) * (alpha_hi - alpha_lo) + alpha_lo
    ).to(torch.float64)
    return alphas


def batch_qsp_loss(
    phi_batch:   torch.Tensor,   # (B, K+1)
    delta_s:     torch.Tensor,   # (B, S)
    alpha_t:     torch.Tensor,   # (B, S)
    Delta_0:     float,
    Omega_max:   float,
    lambda_grad: float = 0.3,
) -> torch.Tensor:
    """
    Vectorized QSP loss over a batch of independently predicted phase vectors.

    Uses build_qsp_unitary_batched to process all B items in a single GPU pass
    (shape: B×S simultaneous 2×2 matmuls) instead of a Python for-loop.
    """
    if lambda_grad > 0.0:
        # Track delta so we can differentiate pred w.r.t. it analytically,
        # avoiding two extra forward passes that finite differences would require.
        delta_in = delta_s.detach().requires_grad_(True)
    else:
        delta_in = delta_s

    U    = build_qsp_unitary_batched(phi_batch, delta_in, Delta_0, Omega_max)  # (B, S, 2, 2)
    pred = U[:, :, 0, 0]                                                        # (B, S) complex
    tgt  = (torch.cos(alpha_t / 2) - 1j * torch.sin(alpha_t / 2)).to(torch.complex128)
    err  = (pred - tgt).abs() ** 2                                              # (B, S) real

    if lambda_grad > 0.0:
        # create_graph=True keeps the second-order graph so d(penalty)/d(phi)
        # flows back through the network during loss.backward().
        dp_real = torch.autograd.grad(pred.real.sum(), delta_in, create_graph=True)[0]
        dp_imag = torch.autograd.grad(pred.imag.sum(), delta_in, create_graph=True)[0]
        err = err + lambda_grad * (dp_real ** 2 + dp_imag ** 2)

    return err.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train_nn(
    cfg:         NNTrainConfig,
    model:       Optional[QSPPhaseNet] = None,
    progress_cb: Optional[Callable[[int, int, float, float], None]] = None,
    verbose:     bool = True,
) -> Tuple[QSPPhaseNet, List[float], List[Tuple[int, float]]]:
    """
    Batch-train QSPPhaseNet given K and Omega (encoded in cfg).

    For each gradient step the training loop:
      1. Samples `batch_size` random α configurations (each of length N).
      2. Encodes them as [cos(αᵢ/2), sin(αᵢ/2)] feature vectors.
      3. Forward-passes the network to get φ predictions.
      4. Samples δ values uniformly inside [δᵢ ± σ] for every peak.
      5. Evaluates batch_qsp_loss (fully differentiable through φ → network).
      6. Back-propagates and updates the network weights.

    Parameters
    ----------
    cfg         : NNTrainConfig
    model       : optionally provide a pre-existing model to continue training
    progress_cb : optional callback(step, total_steps, loss, eta_seconds)
    verbose     : show tqdm progress bar and eval printouts

    Returns
    -------
    model        : trained QSPPhaseNet (best held-out-eval checkpoint)
    train_losses : list of per-step MSE losses
    eval_records : list of (step, eval_loss) tuples at eval checkpoints
    """
    torch.set_default_dtype(torch.float64)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(cfg.out_dir, exist_ok=True)

    if model is None:
        model = QSPPhaseNet(N=cfg.N, K=cfg.K, hidden_dims=cfg.hidden_dims)
    model = model.to(device).double()

    delta_centers = torch.tensor(cfg.delta_vals, dtype=torch.float64, device=device)

    # Fixed held-out evaluation set — same configs for every eval checkpoint
    torch.manual_seed(0)
    eval_alphas = _sample_alpha_configs(
        cfg.eval_configs, cfg.N, cfg.alpha_lo, cfg.alpha_hi, device, cfg.peak_index
    )

    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)

    train_losses: List[float]              = []
    eval_records: List[Tuple[int, float]]  = []
    best_eval_loss = float("inf")
    best_state: Optional[dict]             = None
    start_t = time.perf_counter()

    # ── inner helpers (closures over model / cfg / device) ────────────────────

    def _step(alpha_configs: torch.Tensor) -> float:
        model.train()
        x         = QSPPhaseNet.encode_alphas(alpha_configs)
        phi_batch = model(x)                                   # (B, K+1)
        delta_s, alpha_t = _sample_training_data(
            alpha_configs, delta_centers, cfg.robustness_window,
            cfg.Delta_0, cfg.sample_size, device,
        )
        loss = batch_qsp_loss(
            phi_batch, delta_s, alpha_t,
            cfg.Delta_0, cfg.Omega_max, cfg.lambda_grad,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        sched.step()
        return loss.item()

    @torch.no_grad()
    def _eval() -> float:
        model.eval()
        x         = QSPPhaseNet.encode_alphas(eval_alphas)
        phi_batch = model(x)
        delta_s, alpha_t = _sample_training_data(
            eval_alphas, delta_centers, cfg.robustness_window,
            cfg.Delta_0, cfg.sample_size, device,
        )
        return batch_qsp_loss(
            phi_batch, delta_s, alpha_t, cfg.Delta_0, cfg.Omega_max,
            lambda_grad=cfg.lambda_grad,
        ).item()

    # ── main loop ─────────────────────────────────────────────────────────────

    itr = range(1, cfg.steps + 1)
    if verbose:
        itr = tqdm(itr, desc="NN-QSP training", dynamic_ncols=True)

    ev: float = float("nan")

    for step in itr:
        alpha_configs = _sample_alpha_configs(
            cfg.batch_size, cfg.N, cfg.alpha_lo, cfg.alpha_hi, device, cfg.peak_index
        )

        loss_val = _step(alpha_configs)
        train_losses.append(loss_val)

        if verbose:
            itr.set_postfix({"loss": f"{loss_val:.3e}", "eval": f"{ev:.3e}"})

        if step % cfg.eval_interval == 0:
            ev = _eval()
            eval_records.append((step, ev))
            if verbose:
                tqdm.write(f"  step {step:>7d} | eval_loss = {ev:.4e}")
            if ev < best_eval_loss:
                best_eval_loss = ev
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if step % cfg.checkpoint_interval == 0:
            _save_checkpoint(model, opt, step, loss_val, cfg.out_dir)

        if progress_cb is not None:
            elapsed = time.perf_counter() - start_t
            eta     = (elapsed / step) * (cfg.steps - step)
            progress_cb(step, cfg.steps, loss_val, eta)

    # ── restore best checkpoint and save ──────────────────────────────────────

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    torch.save(model.state_dict(), os.path.join(cfg.out_dir, "model_final.pt"))
    _plot_training_curve(train_losses, eval_records, cfg.out_dir)

    return model, train_losses, eval_records


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_phi(
    model:      QSPPhaseNet,
    alpha_vals: torch.Tensor,   # (N,) or (B, N)
    device:     str = "cpu",
) -> torch.Tensor:
    """
    Predict QSP phase vector(s) for given target rotation angles.

    Parameters
    ----------
    model      : trained QSPPhaseNet
    alpha_vals : (N,) for single config, or (B, N) for a batch
    device     : target device for inference

    Returns
    -------
    (K+1,) for single input, or (B, K+1) for batched input
    """
    model.eval()
    squeeze = alpha_vals.ndim == 1
    a = alpha_vals.unsqueeze(0) if squeeze else alpha_vals
    x = QSPPhaseNet.encode_alphas(a.to(device, dtype=torch.float64))
    phi = model(x)
    return phi.squeeze(0) if squeeze else phi


def evaluate_fidelity(
    model:      QSPPhaseNet,
    alpha_vals: torch.Tensor,   # (N,) target alphas for one configuration
    delta_vals: torch.Tensor,   # (N,) peak detunings (rad/μs)
    cfg_qsp:    TrainConfig,
    device:     str = "cpu",
) -> float:
    """
    Gate fidelity for a single α configuration using the QSP fidelity metric.

    Calls `fidelity` from qsp_fit_x_rotation (5000 samples, uniform window)
    on the phase vector predicted by the model.

    Returns
    -------
    Average gate fidelity in [0, 1].
    """
    phi = predict_phi(model, alpha_vals, device=device).cpu()
    return fidelity(phi, delta_vals.cpu(), alpha_vals.cpu(), cfg_qsp)


def visualize_predictions(
    model:      QSPPhaseNet,
    alpha_vals: torch.Tensor,   # (N,)
    delta_vals: torch.Tensor,   # (N,)
    cfg_qsp:    TrainConfig,
    out_path:   str,
    device:     str = "cpu",
) -> float:
    """
    Predict φ for given α and produce a matrix-element-vs-δ plot.

    Returns the gate fidelity of the predicted phases.
    """
    phi = predict_phi(model, alpha_vals, device=device).cpu()
    fid = plot_matrix_element_vs_delta(phi, cfg_qsp, delta_vals.cpu(), alpha_vals.cpu(), out_path)
    return fid


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_checkpoint(
    model:   nn.Module,
    opt:     torch.optim.Optimizer,
    step:    int,
    loss:    float,
    out_dir: str,
) -> None:
    path = os.path.join(out_dir, f"ckpt_step{step:07d}.pt")
    torch.save({
        "step":        step,
        "loss":        loss,
        "model_state": model.state_dict(),
        "opt_state":   opt.state_dict(),
    }, path)


def _plot_training_curve(
    train_losses: List[float],
    eval_records: List[Tuple[int, float]],
    out_dir:      str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(1, len(train_losses) + 1)
    ax.semilogy(steps, train_losses, alpha=0.5, lw=0.8, label="Train (per step)")
    if eval_records:
        es, el = zip(*eval_records)
        ax.semilogy(es, el, "o-", lw=2, ms=5, label="Eval (held-out)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("QSPPhaseNet – Training Curve")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curve.png"), dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Batch-train QSPPhaseNet: α-encoding → QSP phase vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--K",                 type=int,   default=70,
                    help="QSP order (output has K+1 phases)")
    ap.add_argument("--N",                 type=int,   default=4,
                    help="Number of detuning peaks / target gates")
    ap.add_argument("--Omega_max",         type=float, default=80.0,  metavar="MHz")
    ap.add_argument("--Delta_0",           type=float, default=200.0, metavar="MHz")
    ap.add_argument("--robustness_window", type=float, default=10.0,  metavar="MHz")
    ap.add_argument("--delta_vals",        type=float, nargs="+",
                    default=[-100.0, -32.0, 32.0, 100.0], metavar="MHz")
    ap.add_argument("--alpha_lo",          type=float, default=0.0,
                    help="Lower bound for α sampling (radians)")
    ap.add_argument("--alpha_hi",          type=float, default=4.0 * math.pi,
                    help="Upper bound for α sampling (radians)")
    ap.add_argument("--steps",             type=int,   default=10_000)
    ap.add_argument("--batch_size",        type=int,   default=64)
    ap.add_argument("--sample_size",       type=int,   default=256,
                    help="δ samples per α config per step")
    ap.add_argument("--lr",                type=float, default=1e-3)
    ap.add_argument("--lambda_grad",       type=float, default=0.0,
                    help="Gradient-penalty weight (0 = disabled)")
    ap.add_argument("--device",            type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir",           type=str,   default="nn_qsp_output")
    ap.add_argument("--checkpoint_interval", type=int, default=1000)
    ap.add_argument("--eval_interval",       type=int, default=500)
    ap.add_argument("--peak_index",          type=int, default=None,
                    help="If set, only α at this peak index varies; all others are 0")
    args = ap.parse_args()

    if len(args.delta_vals) != args.N:
        ap.error(f"--delta_vals must have exactly N={args.N} values, "
                 f"got {len(args.delta_vals)}")

    cfg = NNTrainConfig(
        K=args.K,
        N=args.N,
        Omega_max=2 * math.pi * args.Omega_max,
        Delta_0=2 * math.pi * args.Delta_0,
        robustness_window=2 * math.pi * args.robustness_window,
        delta_vals=[2 * math.pi * d for d in args.delta_vals],
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        sample_size=args.sample_size,
        lambda_grad=args.lambda_grad,
        device=args.device,
        out_dir=args.out_dir,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        peak_index=args.peak_index,
    )

    print("=" * 60)
    print("QSPPhaseNet training")
    print("=" * 60)
    print(f"  K={cfg.K}, N={cfg.N}")
    print(f"  Ω_max    = {args.Omega_max} MHz  →  {cfg.Omega_max/(2*math.pi):.2f} × 2π MHz")
    print(f"  Δ₀       = {args.Delta_0} MHz  →  {cfg.Delta_0/(2*math.pi):.2f} × 2π MHz")
    print(f"  σ (win.) = {args.robustness_window} MHz  →  {cfg.robustness_window/(2*math.pi):.2f} × 2π MHz")
    print(f"  δ peaks  = {args.delta_vals} MHz")
    if args.peak_index is not None:
        print(f"  α mode   = single-peak (index {args.peak_index}): "
              f"α_{args.peak_index} ∈ [{args.alpha_lo/math.pi:.3f}π, {args.alpha_hi/math.pi:.3f}π], others = 0")
    else:
        print(f"  α range  = [{args.alpha_lo/math.pi:.3f}π, {args.alpha_hi/math.pi:.3f}π] (all peaks)")
    print(f"  steps={cfg.steps}, batch={cfg.batch_size}, "
          f"sample_size={cfg.sample_size}, lr={cfg.lr}")
    print(f"  device={cfg.device}, out_dir={cfg.out_dir}")
    print("=" * 60)

    model, train_losses, eval_records = train_nn(cfg, verbose=True)

    final_eval = eval_records[-1][1] if eval_records else float("nan")
    print(f"\nDone.  Best eval loss: {min(e for _, e in eval_records):.4e}"
          if eval_records else "\nDone.")
    print(f"Final eval loss : {final_eval:.4e}")
    print(f"Model saved     : {os.path.join(cfg.out_dir, 'model_final.pt')}")
    print(f"Training curve  : {os.path.join(cfg.out_dir, 'training_curve.png')}")

    # Quick demo: predict φ for a representative α configuration
    _demo_base = [0.5 * math.pi, math.pi, 0.3 * math.pi, 1.5 * math.pi]
    if args.peak_index is not None:
        alpha_demo = torch.zeros(cfg.N, dtype=torch.float64)
        alpha_demo[args.peak_index] = _demo_base[args.peak_index % len(_demo_base)]
    else:
        alpha_demo = torch.tensor(_demo_base[:cfg.N], dtype=torch.float64)
    phi_pred = predict_phi(model, alpha_demo, device=cfg.device)
    print(f"\nDemo prediction for α = {[f'{a/math.pi:.2f}π' for a in alpha_demo.tolist()]}:")
    print(f"  φ shape  : {list(phi_pred.shape)}")
    print(f"  φ range  : [{phi_pred.min().item():.3f}, {phi_pred.max().item():.3f}]")

    # Fidelity evaluation for the demo config
    cfg_qsp = TrainConfig(
        Omega_max=cfg.Omega_max,
        Delta_0=cfg.Delta_0,
        robustness_window=cfg.robustness_window,
        K=cfg.K,
        device="cpu",
    )
    delta_t = torch.tensor(cfg.delta_vals, dtype=torch.float64)
    fid = evaluate_fidelity(model, alpha_demo, delta_t, cfg_qsp, device="cpu")
    print(f"  Gate fidelity: {fid:.6f}")


if __name__ == "__main__":
    main()
