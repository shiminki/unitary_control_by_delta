"""
Neural network pulse generator for detuning-selective unitary control.

Given a detuning peak index i, this module trains a neural network that maps
rotation angle theta -> QSP phases phi[0..K], such that the resulting pulse
applies Rx(theta) at peak i and identity at all other peaks.

After training, PCA analysis extracts the effective degrees of freedom and
basis functions b_j(t) with amplitude functions A_j(theta).

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
    ap.add_argument("--hidden_dim", type=int, default=128, help="NN hidden dimension")
    ap.add_argument("--num_layers", type=int, default=4, help="NN hidden layers")
    ap.add_argument("--n_freq", type=int, default=8, help="Fourier input frequencies")
    ap.add_argument("--out_dir", type=str, default="pca_analysis_output", help="Output directory")
    ap.add_argument("--test_only", action="store_true", help="Run tests only")
    ap.add_argument("--skip_tests", action="store_true", help="Skip tests")
    ap.add_argument("--acceptance_test", action="store_true", help="Run acceptance test (slow)")
    ap.add_argument("--M_pca", type=int, default=200, help="PCA theta samples")
    ap.add_argument("--force_retrain", action="store_true", help="Force retraining even if cached model exists")
    args = ap.parse_args()

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
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        force_retrain=args.force_retrain,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating trained network...")
    print("=" * 60)

    theta_eval = torch.linspace(0, 2 * math.pi, 9, dtype=torch.float64)
    results = evaluate_nn(net, theta_eval)
    for tv, fid in zip(results["theta"], results["fidelity"]):
        print(f"  theta = {tv/math.pi:.2f}*pi  |  fidelity = {fid:.6f}")

    # PCA
    print("\n" + "=" * 60)
    print("Running PCA analysis...")
    print("=" * 60)

    pca_result = pca_analysis(net, M=args.M_pca, out_dir=args.out_dir)
    print(f"  Effective DOF: {pca_result['n_effective_dof']}")
    print(f"  Singular values (top 10): {pca_result['singular_values'][:10].round(4)}")
    print(f"  Explained variance (top 10): {pca_result['explained_variance_ratio'][:10].round(4)}")

    # t-SNE of phi trajectory
    print("\nGenerating t-SNE plot...")
    plot_tsne(pca_result["phi_matrix"], pca_result["theta_grid"],
              args.peak_index, args.out_dir)

    # Feature 1: Comparative pulse plots
    compare_thetas = [math.pi / 4, math.pi / 2, math.pi]
    print("\nGenerating comparative pulse plots...")
    plot_comparative_pulses(net, pca_result, compare_thetas, args.out_dir)

    # Feature 2: Comparative matrix element plots (also computes fidelities)
    print("Generating comparative matrix element plots...")
    fidelity_results = plot_comparative_matrix_elements(
        net, pca_result, compare_thetas, args.out_dir)

    # Feature 3: Analytical form (with fidelity comparison)
    print("\n")
    analytical_str = format_analytical_form(pca_result, fidelity_results=fidelity_results)
    print(analytical_str)

    # Save analytical form to file
    analytical_path = os.path.join(args.out_dir, f"analytical_form_peak{args.peak_index}.txt")
    with open(analytical_path, "w") as f:
        f.write(analytical_str)

    print(f"\nResults saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
