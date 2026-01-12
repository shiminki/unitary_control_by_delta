"""
Docstring for eval_phases

Evaluates the fidelity of phases generated from qsp_fit.py
"""


import argparse

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

from qsp_fit import build_U

import matplotlib.pyplot as plt


def fidelity(U_target, U_actual):
    """Compute fidelity between two unitaries."""
    d = U_target.shape[0]
    fid = ( abs(torch.trace(U_target.conj().T @ U_actual)) / d ) ** 2
    return fid


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase_file_dir",
        type=str,
        default="plots/learned_phases_K=100_combine_Q_True.csv",
        help="Directory to load learned phases from.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=torch.pi,
        help="Alpha parameter for target function.",
    )
    args = ap.parse_args()
    phase_file_dir = args.phase_file_dir
    alpha = args.alpha

    I = torch.eye(2)
    R_z_alpha = torch.tensor([
        [np.exp(1j * alpha / 2), 0],
        [0, np.exp(-1j * alpha / 2)]
    ])

    # Load phases
    phi_df = pd.read_csv(phase_file_dir)
    phiset = torch.tensor(phi_df["phi"].values, dtype=torch.float32)

    # Evaluate fidelity
    theta_vals = torch.linspace(-torch.pi, torch.pi, 500)
    U_target = torch.stack([
        I if abs(np.cos(theta)) > 1/np.sqrt(2) else R_z_alpha
     for theta in theta_vals])

    U_actual = build_U(phiset, theta_vals)

    infid_vals = []
    for i in range(len(theta_vals)):
        fid = fidelity(U_target[i], U_actual[i])
        infid_vals.append(max(1 - fid.item(), 1e-12))  # avoid numerical issues


    fig, axs = plt.subplots(3, 1, figsize=(14, 10))

    ax = axs[0]
    ax.plot(theta_vals.numpy(), infid_vals, label="Infidelity")
    ax.set_title(f"Infidelity of QSP Unitaries vs controlled R_z({alpha:.4f})")
    ax.set_xlabel("theta")
    ax.set_ylabel("Infidelity")
    # ax.set_ylim(0, 1.05)
    ax.semilogy()
    ax.grid(True, alpha=0.3)

    # ---- shading of target regions (theta in [-pi, pi]) ----
    t = theta_vals.detach().cpu().numpy()
    tmin, tmax = float(t.min()), float(t.max())

    pi = np.pi

    # helper to clip and shade
    def shade_interval(a, b, label=None, alpha=0.12):
        left = max(a, tmin)
        right = min(b, tmax)
        if right > left:
            ax.axvspan(left, right, alpha=alpha, label=label)

    # |cos(theta)| < 1/sqrt(2)  -> controlled Rz(alpha)
    # intervals within [-pi, pi]: (-3pi/4, -pi/4) and (pi/4, 3pi/4)
    first_rz = True
    for a, b in [(-3 * pi / 4, -pi / 4), (pi / 4, 3 * pi / 4)]:
        shade_interval(a, b, label=r"$R_z(\alpha)$" if first_rz else None, alpha=0.15)
        first_rz = False

    # |cos(theta)| > 1/sqrt(2)  -> Identity
    # complement intervals within [-pi, pi]
    first_I = True
    for a, b in [(-pi, -3 * pi / 4), (-pi / 4, pi / 4), (3 * pi / 4, pi)]:
        shade_interval(a, b, label=r"$I$" if first_I else None, alpha=0.00)
        first_I = False

    ax.legend(loc="best")


    ax = axs[1]
    ax.plot(theta_vals.numpy(), U_actual[:, 0, 0].real.numpy(), label="Re[U00]")
    
    ax.plot(theta_vals.numpy(), U_target[:, 0, 0].real.numpy(), "--", label="Re[Target]")

    ax.set_title("Re[U00] of QSP Unitaries vs Re[Target]")
    ax.set_xlabel("theta")
    ax.set_ylabel("U00")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axs[2]
    ax.plot(theta_vals.numpy(), U_actual[:, 0, 0].imag.numpy(), label="Im[U00]")
    ax.plot(theta_vals.numpy(), U_target[:, 0, 0].imag.numpy(), "--", label="Im[Target]")
    ax.set_title("Im[U00] of QSP Unitaries vs Im[Target]")
    ax.set_xlabel("theta")
    ax.set_ylabel("U00")
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
