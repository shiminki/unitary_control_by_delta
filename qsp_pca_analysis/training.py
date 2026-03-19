"""Training loop and caching for PulseGeneratorNet."""

import math
import os

import torch
from tqdm import tqdm

from single_pulse_optimization_QSP.qsp_fit_x_rotation import build_qsp_unitary

from .constants import (
    DEFAULT_K, DELTA_CENTERS_MHZ, OMEGA_MHZ, DELTA_0_MHZ,
    ROBUSTNESS_WINDOW_MHZ, EPS, EARLY_STOPPING
)
from .model import PulseGeneratorNet


def presample_detunings(
    peak_index: int,
    delta_centers_ang: list,
    robustness_window_ang: float,
    Delta_0_ang: float,
    samples_per_peak: int = 128,
    device: str = "cpu",
) -> tuple:
    """Pre-sample detuning values around each peak (fixed for training).

    Returns
    -------
    delta_all : (N*S,) tensor of detuning values in angular units.
    peak_mask : (N*S,) bool tensor, True for samples at the target peak.
    """
    n_peaks = len(delta_centers_ang)
    delta_list = []
    mask_list = []

    for j in range(n_peaks):
        center = delta_centers_ang[j]
        half_w = robustness_window_ang
        jitter = (2.0 * torch.rand(samples_per_peak, dtype=torch.float64, device=device) - 1.0) * half_w
        delta_s = (center + jitter).clamp(-Delta_0_ang, Delta_0_ang)
        delta_list.append(delta_s)
        mask_list.append(torch.full((samples_per_peak,), j == peak_index, device=device))

    return torch.cat(delta_list), torch.cat(mask_list)


def compute_batch_loss(
    net: PulseGeneratorNet,
    theta_batch: torch.Tensor,
    delta_all: torch.Tensor,
    peak_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute u_00 MSE loss for a batch of target rotation angles.

    Uses pre-sampled, fixed detuning values (no stochastic resampling).
    Computes loss directly from QSP unitary u_00 element—no gradient
    regularization overhead, so only 1 build_qsp_unitary call per theta.

    Parameters
    ----------
    net : PulseGeneratorNet
    theta_batch : (B,) tensor of target rotation angles.
    delta_all : (N*S,) pre-sampled detuning values.
    peak_mask : (N*S,) bool, True for target-peak samples.

    Returns
    -------
    Scalar loss tensor (differentiable).
    """
    B = theta_batch.shape[0]
    device = theta_batch.device

    phi_batch = net(theta_batch)  # (B, K+1)

    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)

    for b in range(B):
        phi_b = phi_batch[b]  # (K+1,)

        # Alpha for target peak = theta_b, for other peaks = 0
        alpha_all = torch.where(
            peak_mask,
            theta_batch[b].expand_as(delta_all),
            torch.zeros_like(delta_all),
        )

        U = build_qsp_unitary(phi_b, delta_all, net.Delta_0_ang, net.Omega_ang)
        pred = U[:, 0, 0]  # (N*S,)
        target = (torch.cos(alpha_all / 2) - 1j * torch.sin(alpha_all / 2)).to(torch.complex128)

        loss_b = ((pred - target).abs() ** 2).mean()
        total_loss = total_loss + loss_b

    return total_loss / B


def train_nn(
    peak_index: int,
    K: int = DEFAULT_K,
    hidden_dim: int = 128,
    num_layers: int = 4,
    n_freq: int = 8,
    steps: int = 5000,
    lr: float = 5e-3,
    batch_size: int = 8,
    samples_per_peak: int = 128,
    delta_centers_mhz: list = None,
    Omega_mhz: float = OMEGA_MHZ,
    Delta_0_mhz: float = DELTA_0_MHZ,
    robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
    device: str = "cpu",
    verbose: bool = True,
    out_dir: str = "nn_pulse_output",
    resample_every: int = 0,
) -> PulseGeneratorNet:
    """Train a PulseGeneratorNet for a given peak index.

    Parameters
    ----------
    resample_every : int
        If > 0, resample detuning points every this many steps (0 = never).

    Returns the trained network.
    """
    if delta_centers_mhz is None:
        delta_centers_mhz = list(DELTA_CENTERS_MHZ)
    torch.set_default_dtype(torch.float64)

    net = PulseGeneratorNet(
        peak_index=peak_index, K=K, hidden_dim=hidden_dim,
        num_layers=num_layers, n_freq=n_freq,
        delta_centers_mhz=delta_centers_mhz,
        Omega_mhz=Omega_mhz, Delta_0_mhz=Delta_0_mhz,
        robustness_window_mhz=robustness_window_mhz,
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Pre-sample detunings once (like GRAPE)
    delta_all, peak_mask = presample_detunings(
        peak_index, net.delta_centers_ang, net.robustness_window_ang,
        net.Delta_0_ang, samples_per_peak, device,
    )

    best_loss = float("inf")
    best_state = None

    iterator = tqdm(range(1, steps + 1), desc=f"Training peak {peak_index}") if verbose else range(1, steps + 1)

    for step in iterator:
        # Optionally resample detunings for generalization
        if resample_every > 0 and step > 1 and step % resample_every == 0:
            delta_all, peak_mask = presample_detunings(
                peak_index, net.delta_centers_ang, net.robustness_window_ang,
                net.Delta_0_ang, samples_per_peak, device,
            )

        # Ensuring it covers theta = 0 and 2pi
        theta_batch = (torch.rand(batch_size, dtype=torch.float64, device=device) * (1 + 2 * EPS) - EPS) \
            * 2 * math.pi
        loss = compute_batch_loss(net, theta_batch, delta_all, peak_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        opt.step()
        sched.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

        if verbose and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{loss_val:.4e}", "best": f"{best_loss:.4e}"})

        if loss_val < EARLY_STOPPING:
            print("Early stopping training")
            break

    if best_state is not None:
        net.load_state_dict(best_state)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(out_dir, f"peak{peak_index}_K{K}.pt"))

    return net


def load_or_train_nn(
    peak_index: int,
    K: int = DEFAULT_K,
    hidden_dim: int = 128,
    num_layers: int = 4,
    n_freq: int = 8,
    steps: int = 5000,
    lr: float = 5e-3,
    batch_size: int = 8,
    samples_per_peak: int = 128,
    delta_centers_mhz: list = None,
    Omega_mhz: float = OMEGA_MHZ,
    Delta_0_mhz: float = DELTA_0_MHZ,
    robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
    device: str = "cpu",
    verbose: bool = True,
    out_dir: str = "nn_pulse_output",
    resample_every: int = 0,
    force_retrain: bool = False,
) -> PulseGeneratorNet:
    """Load cached NN for (peak_index, K) if available, otherwise train.

    Same parameters as train_nn, plus force_retrain to bypass cache.
    """
    cache_path = os.path.join(out_dir, f"peak{peak_index}_K{K}.pt")

    if not force_retrain and os.path.exists(cache_path):
        if verbose:
            print(f"  Loading cached model from {cache_path}")
        if delta_centers_mhz is None:
            delta_centers_mhz = list(DELTA_CENTERS_MHZ)
        net = PulseGeneratorNet(
            peak_index=peak_index, K=K,
            hidden_dim=hidden_dim, num_layers=num_layers, n_freq=n_freq,
            delta_centers_mhz=delta_centers_mhz,
            Omega_mhz=Omega_mhz, Delta_0_mhz=Delta_0_mhz,
            robustness_window_mhz=robustness_window_mhz,
        ).to(device)
        net.load_state_dict(torch.load(cache_path, map_location=device, weights_only=True))
        net.eval()
        return net

    return train_nn(
        peak_index=peak_index, K=K,
        hidden_dim=hidden_dim, num_layers=num_layers, n_freq=n_freq,
        steps=steps, lr=lr, batch_size=batch_size,
        samples_per_peak=samples_per_peak,
        delta_centers_mhz=delta_centers_mhz,
        Omega_mhz=Omega_mhz, Delta_0_mhz=Delta_0_mhz,
        robustness_window_mhz=robustness_window_mhz,
        device=device, verbose=verbose, out_dir=out_dir,
        resample_every=resample_every,
    )
