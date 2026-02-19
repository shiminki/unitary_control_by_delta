"""
Reference:
https://arxiv.org/pdf/2312.08426
"""


from __future__ import annotations

import math
from typing import Callable, Dict, List

import torch

import pandas as pd
import numpy as np
import pwlf

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.colors import TABLEAU_COLORS, to_rgba
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors as mcolors

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import shutil

from qutip import Bloch


# Add to the top of visualize/single_qubit_visualization.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




###############################################################################
# Pauli matrices and helpers – cached per device
###############################################################################

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)

# Simple †‑immortal cache keyed by torch.device.
_PAULI_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_paulis(device: torch.device) -> torch.Tensor:
    """Return a stack ``(4, 2, 2)`` of *(I, σₓ, σ_y, σ_z)* on *device*.

    The tensors are created on their first use on each device and then reused
    to avoid needless kernel launches and host‑to‑device traffic.
    """
    if device not in _PAULI_CACHE:
        _PAULI_CACHE[device] = torch.stack(
            [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU], dim=0
        ).to(device)
    return _PAULI_CACHE[device]


########################
# Helper Functions ######
#########################

def fidelity(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int) -> torch.Tensor:
    """Entanglement fidelity F = (|Tr(U_out^† U_target)|² + d) / d(d + 1)."""
    # trace over last two dims, keep batch
    # Batched conjugate transpose and matrix multiplication
    U_out_dagger = U_out.conj().transpose(-1, -2)  # [batch, 2, 2]
    product = U_out_dagger @ U_target  # [batch, 2, 2]

    # print(product, product.shape)

    # Batched trace calculation
    trace = torch.einsum('bii->b', product)  # [batch]
    trace_squared = torch.abs(trace) ** 2

    d = 2 ** num_qubits

    return (trace_squared + d) / (d * (d + 1))


def get_ore_ple_error_distribution(batch_size:int, delta_std: float = 1.0, epsilon_std: float=0.05) -> torch.tensor:
    ore_error = torch.randn(batch_size) * delta_std
    ple_error = torch.randn(batch_size) * epsilon_std
    return torch.stack([ore_error, ple_error])


def batched_unitary_generator(
    pulses: torch.Tensor,
    error: torch.Tensor,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 2)``, where each pulse is
        ``[Δ, Ω, φ, t]`` (detuning, Rabi amplitude, phase, duration).
    error : torch.Tensor
        Shape ``(2, B,)`` static off‑resonant detuning and pulse length error for each
        batch element.  If you fuse Monte‑Carlo repeats into the batch, just
        expand ``delta`` accordingly.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 2, 2)`` complex64/128 – the composite unitary ``U_L ⋯ U_1``.
    """

    if pulses.ndim != 3 or pulses.shape[-1] != 2:
        raise ValueError("'pulses' must have shape (B, L, 2)")

    B, L, _ = pulses.shape
    device = pulses.device
    dtype = torch.cfloat

    # Unpack and reshape to broadcast with Pauli matrices.
    phi, tau = pulses.unbind(dim=-1)  # each (B, L)

    # (4, 2, 2) on correct device
    pauli = _get_paulis(device).type(dtype)

    # ORE and PLE
    delta = error[0]
    epsilon = error[1]

    # Build base Hamiltonian H₀ for every pulse in parallel.
    H_base = (
        torch.cos(phi)[..., None, None] * pauli[1]
        + torch.sin(phi)[..., None, None] * pauli[2]
    )
    
    H = H_base + delta[..., None, None, None] * pauli[3]

    H = 0.5 * H * (1 + epsilon[..., None, None, None])

    # U_k = exp(-i H_k t_k)
    U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 2, 2)


    # U: (B, L, 2, 2)   want: U[:, L-1] @ ... @ U[:, 1] @ U[:, 0]
    X = U
    I = torch.eye(2, dtype=dtype, device=device).expand(B, 1, 2, 2)

    while X.size(1) > 1:
        # pad to even length
        if (X.size(1) & 1) == 1:
            X = torch.cat([X, I], dim=1)
        # pairwise multiply preserving left-to-right order:
        # (U1 @ U0), (U3 @ U2), ...
        X = X[:, 1::2] @ X[:, 0::2]

    U_out = X[:, 0]  # (B, 2, 2)


    return U_out


#######################
# SCORE Dataset #######
#######################


angle_vec_dict = {
    1/4 : [1.34820, 1.32669, 1.77042, 2.16800],
    1/3 : [1.41901, 1.35864, 1.77664, 2.13759],
    1/2 : [1.55280, 1.42267, 1.78586, 2.07559],
    2/3 : [1.67478, 1.47865, 1.78919, 2.02043],
    3/4 : [1.73053, 1.49972, 1.78853, 1.99939],
    1   : [1.87342, 1.52524, 1.78436, 1.97330]
}

unitaries = {
    "X(pi)" : [(1, 0)],
    "X(pi-2)" : [(1/2, 0)],
    "Hadamard" : [(1, 0), (1/2, 1/2)],
    "Z(pi-4)" : [(1, 0), (1/2, 1/2), (1/4, 0), (1, 0), (1/2, 1/2)]
}


def SCOREn_config(n, phi):
    angle_vec = angle_vec_dict[n]
    config = []
    Angle = np.pi * n

    for i, angle in enumerate(angle_vec):
        config.append({
            "phi": torch.tensor([phi + (i % 2) * np.pi], dtype=torch.complex64),
            "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
            "angle": torch.tensor([angle * np.pi], dtype=torch.complex64)
        })
        Angle += (-1)**(len(angle_vec) - 1 - i) * 2 * angle * np.pi

    config.append({
        "phi": torch.tensor([phi], dtype=torch.complex64),
        "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
        "angle": torch.tensor([Angle], dtype=torch.complex64)
    })

    for i, angle in reversed(list(enumerate(angle_vec))):
        config.append({
            "phi": torch.tensor([phi + (i % 2) * np.pi], dtype=torch.complex64),
            "theta": torch.tensor([np.pi/2], dtype=torch.complex64),
            "angle": torch.tensor([angle * np.pi], dtype=torch.complex64)
        })

    # config_to_tensor = [
    #     [0, 1, x["phi"], x["angle"]]
    #     for x in config
    # ]
    # config_to_tensor = [
    #     [x["phi"], x["angle"]]
    #     for x in config
    # ]

    dt = sum(x["angle"] for x in config) / 400

    config_to_tensor = []

    for x in config:
        phi = x["phi"].to(torch.float)
        angle = x["angle"].to(torch.float)
        N = math.ceil(angle / dt)
        for _ in range(N):
            config_to_tensor.append(
                [phi, angle / N]
            )

    return torch.tensor(config_to_tensor)


def build_SCORE_pulses(SCORE_emb=False):
    SCORE_pulses = []

    if SCORE_emb:
        unitaries = {
            angle : [(angle, 0)]
            for angle in angle_vec_dict
        }
    save_dir = "weights/SCORE_Pulse/"
    os.makedirs(save_dir, exist_ok=True)

    for target in unitaries:
        pulses = []

        for n, phi in reversed(unitaries[target]):
            pulses.append(SCOREn_config(n, phi * np.pi))
        
        SCORE_pulses.append(torch.stack(pulses))
        x = SCORE_pulses[-1]
        x = x.reshape(-1, x.shape[2]) 
        df = pd.DataFrame(x.to(torch.float))
        if type(target) is float:
            target = str(np.round(target, 2))
        df.to_csv(os.path.join(save_dir, f"{target}_SCORE_pulse.csv"), index=False)

    SCORE_pulses = [x.reshape(-1, x.shape[-1]) for x in SCORE_pulses]

    

    torch.save(SCORE_pulses, os.path.join(save_dir, "SCORE_pulse.pt"))

    return SCORE_pulses


#############################
# Pulse Parameter Plot ######
#############################


def plot_pulse_param(file_path, title, df, Omega):
    t = df["t (us)"].to_numpy(dtype=float) * 1000   # duration in ns
    H_x = df["Omega_x (2pi MHz)"].to_numpy(dtype=float)   # H_x in 2pi MHz
    H_y = df["Omega_y (2pi MHz)"].to_numpy(dtype=float)   # H_y in 2pi MHz
    H_z = df["Omega_z (2pi MHz)"].to_numpy(dtype=float)   # H_z in 2pi MHz


    # Precompute edges in units of pi for plotting
    t_edges = np.concatenate(([0.0], np.cumsum(t)))  # length N+1


    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(10, 6),
        sharex=True,
    )

    axes[0].step(t_edges, np.r_[H_x, H_x[-1]], where='post')
    axes[0].set_ylabel(r"$\Omega_x(t) (2\pi$ MHz)")
    axes[0].grid(True)
    axes[0].set_ylim(-1.1 * Omega, 1.1 * Omega)

    axes[1].step(t_edges, np.r_[H_y, H_y[-1]], where='post')
    axes[1].set_ylabel(r"$\Omega_y(t) (2\pi$ MHz)")
    axes[1].grid(True)
    axes[1].set_ylim(-1.1 * Omega, 1.1 * Omega )

    axes[2].step(t_edges, np.r_[H_z, H_z[-1]], where='post')
    axes[2].set_ylabel(r"$\Omega_z(t) (2\pi$ MHz)")
    axes[2].grid(True)
    axes[2].set_ylim(-10, 200)
    axes[2].set_xlabel(r"Rotation time (ns)")

    

    # --- Title & save ---
    total_time = sum(t) 
    fig.suptitle(f"Optimal Control for {title}\nRabi = {Omega} (2pi MHz), duration = {total_time:.2f} ns", fontsize=16)
    os.makedirs(file_path, exist_ok=True)
    out_path = os.path.join(file_path, f"{title}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path)
    plt.close(fig)



#############################
# Fideltiy Contour Plot #####
#############################


def fidelity_contour_plot(target_name, U_target, pulse, name, save_dir, M=10000, phase_only=True, Omega=None):

    # print(pulse[:-1].to(dtype=torch.float64))
    total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / np.pi

    errors_mc = get_ore_ple_error_distribution(M, 1, 0.05)
    U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
    pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)

    g = batched_unitary_generator

    U_out_plot = g(pulses_plot, errors_mc)

    print(U_target_plot.shape, U_out_plot.shape, pulses_plot.shape)
    F = fidelity(U_out_plot, U_target_plot, 1)


    F_mean = F.mean().item()
    F_err = F.std().item() / np.sqrt(M)

    
    # Create 2D grid of ORE and PLE
    ORE_vals = torch.linspace(-3, 3, 1000)
    PLE_vals = torch.linspace(-0.15, 0.15, 50)
    ORE_grid, PLE_grid = torch.meshgrid(ORE_vals, PLE_vals, indexing="ij")

    # Flatten the grid for batch processing
    ORE_flat = ORE_grid.flatten()
    PLE_flat = PLE_grid.flatten()

    # Shape: (2, N)
    errors_grid = torch.stack([ORE_flat, PLE_flat], dim=0)

    # Repeat target and pulse
    N = errors_grid.shape[1]
    U_target_grid = U_target.expand(N, -1, -1)  # (N, d, d)
    pulses_grid = pulse.expand(N, -1, -1)       # (N, L, P)

    # Evaluate fidelity
    U_out_grid = g(pulses_grid, errors_grid)
    F_grid = fidelity(U_out_grid, U_target_grid, 1).reshape(1000, 50)

    # Convert meshgrid to numpy
    ORE_np = ORE_grid.detach().cpu().numpy()
    PLE_np = PLE_grid.detach().cpu().numpy()
    F_np = F_grid.numpy()

    title = f"{target_name} of {name}\nE[F] = {F_mean:.4f} +/- {F_err:.4f}\nTotal Evolution Time: {total_time:.2f} pi"
    x_label = r"$\delta / \Omega_{\max} \sim N(0, 1)$"
    y_label = r"$\epsilon / \Omega_{\max} \sim N(0, 0.05^2)$"


    if Omega is not None:
        Omega_angular = 2 * math.pi * Omega
        total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / Omega_angular
        title = f"{target_name} of {name}\nE[F] = {F_mean:.4f} +/- {F_err:.4f}\nTotal Evolution Time: {(total_time * 1000):.2f} ns ({(total_time * Omega * 2):.2f} pi) with Rabi Freq {Omega} (2pi MHz)"

        ORE_np *= Omega
        PLE_np *= Omega

        x_label = r"$\delta$ (MHz, scaled by Rabi freq)"
        y_label = r"$\epsilon$ (MHz, scaled by Rabi freq)"

    plt.figure(figsize=(8, 6))
    # contour = plt.contourf(ORE_np, PLE_np, F_np, levels=20, cmap='viridis')
    contour = plt.contourf(ORE_np, PLE_np, F_np, levels=[0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.999, 1.0], cmap='viridis')
    plt.contour(ORE_np, PLE_np, F_np, levels=[0.95, 0.99, 0.999], colors='white', linewidths=1.5)

    plt.colorbar(contour, label='Fidelity')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(
        os.path.join(save_dir, f"{target_name}.png")
    )



######################################
# Average Fidelity by std(delta) #####
######################################


def get_avg_fidelity(U_target, pulse, M=10000, phase_only=True, delta_list=None):

    fidelities = {}
    g = batched_unitary_generator

    if delta_list is None:
        delta_list = [0.1 * (i + 1) for i in range(10)]

    for delta_std in tqdm(delta_list):
        errors_mc = get_ore_ple_error_distribution(M, delta_std, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)


        U_out_plot = g(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        fidelities[delta_std] = f"{F_mean:.4f} +/- {F_err:.4f}"
    
    return fidelities


def plot_fidelity_by_std(target_name, U_target, pulse, name, save_dir, M=10000, phase_only=True):

    # print(pulse[:-1].to(dtype=torch.float64))
    total_time = sum(pulse[:, -1].to(dtype=torch.float64)) / np.pi

    fidelities = {}
    g = batched_unitary_generator

    delta_vals = torch.arange(0.01, 2.0, 0.01)

    for delta_std in tqdm(delta_vals):
        errors_mc = get_ore_ple_error_distribution(M, delta_std, 0.05)
        U_target_plot = torch.stack([U_target]).repeat_interleave(M, dim=0)
        pulses_plot = torch.stack([pulse]).repeat_interleave(M, dim=0)

        U_out_plot = g(pulses_plot, errors_mc)
        F = fidelity(U_out_plot, U_target_plot, 1)

        F_mean = F.mean().item()
        F_err = F.std().item() / np.sqrt(M)

        fidelities[delta_std] = (F_mean, F_err)

        
    # Extract values for plotting
    delta_vals = list(fidelities.keys())
    F_means = [fidelities[d][0] for d in delta_vals]
    inF_means = [1 - fidelities[d][0] for d in delta_vals]
    F_errs = [fidelities[d][1] for d in delta_vals]

    # Piece-wise Linear Fit
    num_segments = 2
    model = pwlf.PiecewiseLinFit(delta_vals, F_means)
    breaks = model.fit(num_segments)
    # Get slopes and intercepts
    slopes = model.slopes
    intercepts = model.intercepts
    F_pred = model.predict(delta_vals)

    # Plotting fidelity vs delta_std
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, F_means, yerr=F_errs, fmt='o-', capsize=4)
    plt.plot(delta_vals, F_pred, 'r--', label=f'Piecewise Linear Fit ({num_segments} segments)')

    # Annotate equations
    for i in range(num_segments):
        x_start = breaks[i]
        x_end = breaks[i+1]
        mid_x = (x_start + x_end) / 2
        mid_y = model.predict([mid_x])[0]
        
        eqn = f"y = {slopes[i]:.3f}x + {intercepts[i]:.3f}"
        plt.text(mid_x, mid_y - 0.03, eqn, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))


    plt.xlabel(r"Std$(\delta / \Omega_{\max})$")
    plt.ylabel("Expected Fidelity")
    plt.title(
        f"Fidelity curve for {target_name} of {name}\n"
        f"Total Evolution Time: {total_time:.2f} π"
    )
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(0.6, 1)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"{target_name}_fidelity.png")
    )

    # Piece-wise Linear Fit
    num_segments = 3
    log_deltas = np.log(delta_vals)
    log_inF = np.log(inF_means)
    model = pwlf.PiecewiseLinFit(log_deltas, log_inF)
    breaks = model.fit(num_segments)
    # Get slopes and intercepts
    slopes = model.slopes
    intercepts = model.intercepts
    inF_pred = np.exp(model.predict(log_deltas))

    # Plotting log(infidelity) vs delta_std
    plt.figure(figsize=(8, 6))
    plt.errorbar(delta_vals, inF_means, yerr=F_errs, fmt='o-', capsize=4)
    plt.plot(delta_vals, inF_pred, 'r--', label=f'Piecewise Linear Fit ({num_segments} segments)')

    plt.semilogy()
    plt.semilogx()
    # Annotate equations

    for i in range(num_segments):
        x_start = breaks[i]
        x_end = breaks[i+1]
        mid_x_log = (x_start + x_end) / 2
        mid_x = np.exp(mid_x_log)  # Transform back to delta_vals space
        mid_y = np.exp(model.predict([mid_x_log])[0])  # Also inverse-transform y

        eqn = f"log(y) = {slopes[i]:.3f} log(x) + {intercepts[i]:.3f}"
        plt.text(mid_x, mid_y * 1.2, eqn, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.xlabel(r"Std$(\delta / \Omega_{\max})$")
    plt.ylabel("Expected Infidelity")
    plt.title(
        f"Infidelity curve for {target_name} of {name}\n"
        f"Total Evolution Time: {total_time:.2f} π"
    )
    
    plt.grid(True)
    plt.tight_layout()

    plt.ylim(1e-3, 1)
    plt.savefig(
        os.path.join(save_dir, f"{target_name}_infidelity_with_fit.png")
    )


######################################
# Qubit Ensemble Evolution Video #####
######################################



def animate_multi_error_bloch(
    bloch_vectors_list,   # list of arrays (T x 3)
    pulse_info_list,      # list of lists of pulse tuples
    fidelity_list,        # list of final fidelities
    delta_list,           # list of delta values
    epsilon_list,         # list of epsilon values
    name,
    save_path="multi_bloch_qutip.mp4",
    phase_only=True,
    Omega=None,
    label_list=None,      # optional per-qubit legend labels (overrides default)
):
    num_qubits = len(bloch_vectors_list)
    num_frames = bloch_vectors_list[0].shape[0]

    # Prepare colors and legend handles
    colors = list(TABLEAU_COLORS.values())[:num_qubits + 1]

    legend_handles = []
    for idx in range(num_qubits):
        if label_list is not None:
            lbl = label_list[idx]
        else:
            j_del = idx % len(delta_list)
            lbl = fr"$\delta$={delta_list[j_del]:.2f}, F={fidelity_list[idx]:.4f}"
        legend_handles.append(Line2D([0], [0], color=colors[idx % 10], lw=2, label=lbl))

    # Create figure and axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(name, fontsize=14)

    # Initialize Bloch object with our axes
    b = Bloch(fig=fig, axes=ax)
    b.view = [20, 45]
    # We'll override colors per trajectory, so set defaults to unused
    b.vector_color = colors
    b.point_color = colors

    # choose which index is τ (phase_only tuples are (something, φ, τ), otherwise (…, τ) at position 4)
    tau_idx = 2 if phase_only else 4

    # build an array: step_times[k] = total τ across all qubits at frame k
    step_times = []
    for k in range(num_frames):
        tot = 0.0
        for i in range(num_qubits):
            if k < len(pulse_info_list[i]):
                tot += pulse_info_list[i][k][tau_idx]
        step_times.append(tot / num_qubits)

    # cumulative time (in units of π)
    cumulative_times = np.cumsum(step_times) / np.pi

    
    def update(frame):
        # Clear previous frame
        b.clear()

        # Plot trajectories and vectors
        for i in range(num_qubits):
            traj = bloch_vectors_list[i][: frame + 1]
            xs, ys, zs = traj[:, 0].tolist(), traj[:, 1].tolist(), traj[:, 2].tolist()
            # trajectory line
            b.add_points([xs, ys, zs], meth='l', colors=[to_rgba(colors[i % 10]) for _ in range(3)], alpha=0.5)
            # head of trajectory
            vec = bloch_vectors_list[i][frame].tolist()

            b.add_vectors([vec], colors=colors[i % 10])
            # Hamiltonian arrow
            if pulse_info_list and frame < len(pulse_info_list[i]):
                if not phase_only:
                    _, D, O, phi, tau = pulse_info_list[i][frame]
                    ham_vec = [O * np.cos(phi), O * np.sin(phi), D]
                    b.add_vectors([ham_vec])
                else:
                    _, phi, tau = pulse_info_list[i][frame]
                    # ham_vec = [np.cos(phi), np.sin(phi), 0]
                
        T = cumulative_times[frame]
        if Omega is not None:
            title_str = (
                f"{name}\n"
                fr"Total Time: {(1000 * T / (2 * Omega)):.4f} ns ({T:.4f}$\pi)$"
                f"\nE[F] = {np.mean(fidelity_list):.4f} +/- {(np.std(fidelity_list) / np.sqrt(len(fidelity_list))):.4f}"
            )
        else:
            title_str = (
                f"{name}\n"
                fr"Total Time: {T:.4f}$\pi$"
                f"\nE[F] = {np.mean(fidelity_list):.4f} +/- {(np.std(fidelity_list) / np.sqrt(len(fidelity_list))):.4f}"
            )
        fig.suptitle(title_str, fontsize=14)
        # Draw sphere and elements
        b.make_sphere()
        b.render()
        # Add legend
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.05, 1.0), fontsize=8)

    # Create animation
    ani = FuncAnimation(fig, update, frames=num_frames, interval=50)
    # Save and close
    ani.save(save_path, fps=15, dpi=150)
    plt.close(fig)




# def animate_multi_error_bloch(
#     bloch_vectors_list,      # list of (P x 3) arrays – one row per pulse end‑point
#     pulse_info_list,         # list[ list[pulse_tuple] ]
#     fidelity_list,           # list[float]
#     delta_list,              # list[float]
#     epsilon_list,            # list[float]
#     name: str,
#     save_path="multi_bloch_qutip.mp4",
#     phase_only=True,
#     time_subdiv: float=0.2,    # ↑  set to 1 → Δt = min τ ; 2 → half that; etc.
# ):
#     """
#     Make a movie in which *each* frame stands for a fixed span Δt (same for
#     every qubit and every pulse).  The smallest pulse duration in the whole
#     experiment is used as the base time‑step and can be subdivided further
#     with `time_subdiv`.

#     Parameters
#     ----------
#     time_subdiv : int, optional
#         Bigger value → finer time resolution (and more frames).  For example,
#         time_subdiv = 2 means Δt = min(τ) / 2.
#     """
#     # ------------------------------------------------------------------
#     # ❶ Pre‑compute some constants we will need many times
#     # ------------------------------------------------------------------
#     num_qubits     = len(bloch_vectors_list)
#     num_pulses     = bloch_vectors_list[0].shape[0]          # old “frames”
#     tau_idx        = 2 if phase_only else 4                  # where τ lives

#     # pulse k → average τ_k  (same length we used before)
#     step_times = [
#         np.mean([pulse_info_list[i][k][tau_idx] for i in range(num_qubits)])
#         for k in range(num_pulses - 1)
#     ]

#     # smallest pulse duration → our base Δt  (optionally subdivided)
#     non_zero_times = [t for t in step_times if t > 0]
#     if not non_zero_times:
#         raise ValueError("All pulse durations are zero—nothing to animate.")
#     dt = min(non_zero_times) / time_subdiv        # <-- now float‑safe
#     # -------------------------------------------------------------

#     # how many frames does pulse k contribute?
#     frames_per_pulse = [
#         max(1, int(round(t / dt))) for t in step_times
#     ]
#     # total number of *new* frames after resampling
#     total_frames = sum(frames_per_pulse)

#     # map “new frame index f  →  pulse index k”
#     pulse_idx_for_frame = np.concatenate([
#         np.full(nf, k, dtype=int) for k, nf in enumerate(frames_per_pulse)
#     ])

#     # ------------------------------------------------------------------
#     # ❷ Build resampled Bloch trajectories (same length for all qubits)
#     # ------------------------------------------------------------------
#     new_bloch_vectors_list = []
#     for i in range(num_qubits):
#         traj = []
#         for k, n_frames in enumerate(frames_per_pulse):
#             traj.extend([bloch_vectors_list[i][k]] * n_frames)   # repeat
#         new_bloch_vectors_list.append(np.vstack(traj))           # (T' x 3)

#     # ------------------------------------------------------------------
#     # ❸ Figure & Bloch initialisation  (unchanged except for colours)
#     # ------------------------------------------------------------------
#     norm   = mcolors.Normalize(vmin=min(delta_list), vmax=max(delta_list))
#     cmap   = cm.get_cmap('viridis')
#     colors = [cmap(norm(d)) for d in delta_list]

#     fig = plt.figure(figsize=(8, 8))
#     ax  = fig.add_subplot(111, projection='3d')
#     fig.suptitle(name, fontsize=14)

#     sm = cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array(delta_list)
#     cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
#     cbar.set_label(r'Detuning $\delta$')

#     b = Bloch(fig=fig, axes=ax)
#     b.view          = [20, 45]
#     b.vector_color  = colors
#     b.point_color   = colors

#     # ------------------------------------------------------------------
#     # ❹ Animation callback
#     # ------------------------------------------------------------------
#     def update(frame: int):
#         b.clear()

#         # draw trajectories up to *this* resampled frame
#         for i in range(num_qubits):
#             traj = new_bloch_vectors_list[i][:frame + 1]
#             xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
#             b.add_points([xs, ys, zs], meth='l',
#                          colors=[colors[i % num_qubits]] * len(xs), alpha=0.5)

#             vec = new_bloch_vectors_list[i][frame]
#             b.add_vectors([vec], colors=[colors[i % num_qubits]])

#             # show current control Hamiltonian arrow
#             k = pulse_idx_for_frame[frame]               # active pulse
#             if not phase_only:
#                 _, D, O, phi, _ = pulse_info_list[i][k]
#                 ham_vec = [O * np.cos(phi), O * np.sin(phi), D]
#                 b.add_vectors([ham_vec])
#             else:
#                 # phase‑only → arrow lies in x‑y plane
#                 _, phi, _ = pulse_info_list[i][k]
#                 ham_vec = [np.cos(phi), np.sin(phi), 0]
#                 b.add_vectors([ham_vec])

#         # title with running time (π‑units) & fidelity summary
#         T_pi   = (frame + 1) * dt / np.pi       # time elapsed in units of π
#         F_mean = np.mean(fidelity_list)
#         F_err  = np.std(fidelity_list) / np.sqrt(len(fidelity_list))
#         fig.suptitle(
#             f"{name}\n"
#             fr"Total Time: {T_pi:.4f}$\pi$"
#             f"\nExpected Fidelity: {F_mean:.4f} ± {F_err:.4f}",
#             fontsize=14,
#         )

#         b.make_sphere()
#         b.render()
        

#     # ------------------------------------------------------------------
#     # ❺ Build & save movie
#     # ------------------------------------------------------------------
#     ani = FuncAnimation(fig, update, frames=total_frames, interval=50)

#     ffmpeg_path = shutil.which("ffmpeg")
#     if ffmpeg_path:
#         writer = FFMpegWriter(fps=max(1, total_frames // 10), codec="libx264")
#     else:
#         save_path = os.path.splitext(save_path)[0] + ".mp4"
#         writer    = PillowWriter(fps=max(1, total_frames // 10))

#     ani.save(save_path, writer=writer, dpi=150)
#     plt.close(fig)
