"""
Neural network pulse generator for detuning-selective unitary control.

Given a detuning peak index i, this module trains a neural network that maps
rotation angle alpha -> QSP phases phi[0..K], such that the resulting pulse
applies Rx(alpha) at peak i and identity at all other peaks.

After training, PCA analysis extracts the effective linear components and
basis functions b_j(t) with amplitude functions A_j(alpha).

Usage:
    # Run tests first, then train for peak 0 and do PCA
    python nn_pulse_generator.py --peak_index 0 --steps 5000

    # Run only tests
    python nn_pulse_generator.py --test_only

    # Run acceptance test (slow, ~3 min)
    python nn_pulse_generator.py --test_only --acceptance_test

    # Use cached model (skip retraining if saved weights exist)
    python nn_pulse_generator.py --peak_index 0

    # Force retraining even if cache exists
    python nn_pulse_generator.py --peak_index 0 --force_retrain
"""

import argparse
import math
import os
import sys

import torch

from qsp_pca_analysis import (
    DEFAULT_K,
    load_or_train_nn,
    evaluate_nn,
    pca_analysis,
    plot_tsne,
    plot_comparative_pulses,
    plot_comparative_matrix_elements,
    format_analytical_form,
    run_tests,
    run_acceptance_test,
)


def main():
    ap = argparse.ArgumentParser(description="NN pulse generator for QSP")
    ap.add_argument("--peak_index", type=int, default=0, help="Detuning peak index (0..3)")
    ap.add_argument("--K", type=int, default=DEFAULT_K, help="QSP degree")
    ap.add_argument("--steps", type=int, default=8000, help="Training steps")
    ap.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    ap.add_argument("--batch_size", type=int, default=100, help="Theta batch size")
    ap.add_argument("--hidden_dim", type=int, default=256, help="NN hidden dimension")
    ap.add_argument("--num_layers", type=int, default=8, help="NN hidden layers")
    ap.add_argument("--n_freq", type=int, default=8, help="Fourier input frequencies")
    ap.add_argument("--out_dir", type=str, default="pca_analysis_output", help="Output directory")
    ap.add_argument("--test_only", action="store_true", help="Run tests only")
    ap.add_argument("--skip_tests", action="store_true", help="Skip tests")
    ap.add_argument("--acceptance_test", action="store_true", help="Run acceptance test (slow)")
    ap.add_argument("--M_pca", type=int, default=200, help="PCA theta samples")
    ap.add_argument("--force_retrain", action="store_true", help="Force retraining even if cached model exists")
    ap.add_argument("--period_in_pi", type=int, default=4, choices=[2, 4],
                    help="Training/analysis period: 2 for alpha in [0,2pi), 4 for [0,4pi)")
    ap.add_argument("--device", type=str, default="auto",
                    help="Device for training: 'cpu', 'cuda', or 'auto' (default: auto-detect)")
    args = ap.parse_args()

    alpha_max = args.period_in_pi * math.pi
    polyfit_max_deg = 5 if args.period_in_pi == 2 else 10
    actual_out_dir = os.path.join(args.out_dir, f"{args.period_in_pi}pi")

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Auto-scale batch_size on GPU to fill ~70 GB target.
    # In eager mode (no torch.compile), autograd saves ~16 float64 tensors
    # of size (batch_size * D) per K iteration for the backward pass.
    # Measured: ~193 bytes per (batch_size * D) per K iteration.
    batch_size = args.batch_size
    if device != "cpu" and args.batch_size == 100:  # only auto-scale if user didn't override
        D = 4 * 128  # N_peaks * samples_per_peak
        target_bytes = 70 * (1024 ** 3)  # aim for ~70 GB
        bytes_per_bd_per_k = 200  # ~16 tensors * 8 bytes * ~1.5 overhead
        max_bd = int(target_bytes / (bytes_per_bd_per_k * args.K))
        batch_size = max(100, max_bd // D)
        print(f"  Auto batch_size={batch_size} (K={args.K}, target=70GB)")

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    if not args.skip_tests:
        print("=" * 60)
        print("Running unit tests...")
        print("=" * 60)
        if not run_tests():
            print("Unit tests FAILED. Aborting.")
            sys.exit(1)
        print("\nAll unit tests passed.\n")

    if args.acceptance_test:
        print("=" * 60)
        print("Running acceptance test (~3 min)...")
        print("=" * 60)
        if not run_acceptance_test():
            print("Acceptance test FAILED.")
            sys.exit(1)
        print("\nAcceptance test passed.\n")

    if args.test_only:
        return

    # Train (or load cached)
    print("=" * 60)
    print(f"Training NN for peak {args.peak_index}, K={args.K}, steps={args.steps}")
    print("=" * 60)

    net = load_or_train_nn(
        peak_index=args.peak_index,
        K=args.K,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_freq=args.n_freq,
        steps=args.steps,
        lr=args.lr,
        batch_size=batch_size,
        out_dir=actual_out_dir,
        force_retrain=args.force_retrain,
        alpha_max=alpha_max,
        device=device,
    )

    # Move to CPU for evaluation/PCA (uses numpy internally)
    net = net.cpu()
    net.eval()

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating trained network...")
    print("=" * 60)

    alpha_eval = torch.linspace(0, alpha_max, 9, dtype=torch.float64)
    results = evaluate_nn(net, alpha_eval)
    for av, fid in zip(results["alpha"], results["fidelity"]):
        print(f"  alpha = {av/math.pi:.2f}*pi  |  fidelity = {fid:.6f}")

    # PCA
    print("\n" + "=" * 60)
    print("Running PCA analysis...")
    print("=" * 60)

    pca_result = pca_analysis(net, M=args.M_pca, out_dir=actual_out_dir,
                              alpha_max=alpha_max, polyfit_max_deg=polyfit_max_deg)
    print(f"  Effective linear components: {pca_result['n_effective_dof']}")
    print(f"  Singular values (top 10): {pca_result['singular_values'][:10].round(4)}")
    print(f"  Explained variance (top 10): {pca_result['explained_variance_ratio'][:10].round(4)}")

    # Comparative plots: for each fit type, generate full-basis and 2-basis comparisons
    compare_alphas = [math.pi / 2, math.pi, 3 * math.pi / 2]
    n_basis = pca_result["n_effective_dof"]

    for fit_type in ["fourier_fit", "polyfit"]:
        fit_dir = os.path.join(actual_out_dir, fit_type)

        for n_basis_label, n_basis_val in [("full", n_basis), ("2 basis", 2)]:
            sub_dir = os.path.join(fit_dir, n_basis_label)

            print(f"\nGenerating comparative plots ({fit_type}, {n_basis_label}, d={n_basis_val})...")

            plot_comparative_pulses(net, pca_result, compare_alphas, sub_dir,
                                    fit_type=fit_type, n_basis=n_basis_val)
            fidelity_results = plot_comparative_matrix_elements(
                net, pca_result, compare_alphas, sub_dir,
                fit_type=fit_type, n_basis=n_basis_val)

            analytical_str = format_analytical_form(
                pca_result, fit_type=fit_type, fidelity_results=fidelity_results)
            if fit_type == "fourier_fit" and n_basis_label == "full":
                print(analytical_str)
            os.makedirs(sub_dir, exist_ok=True)
            analytical_path = os.path.join(
                sub_dir, f"analytical_form_peak{args.peak_index}.txt")
            with open(analytical_path, "w") as f:
                f.write(analytical_str)

    print(f"\nResults saved to {actual_out_dir}/")

if __name__ == "__main__":
    main()
