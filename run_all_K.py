"""Run PCA analysis for multiple K values sequentially.

Usage (Colab A100):
    python run_all_K.py --out_dir /path/to/output

Each K value runs one at a time, using the full GPU memory.
"""

import argparse
import os
import subprocess
import sys

K_VALUES = [100, 150, 200, 300]


def main():
    ap = argparse.ArgumentParser(description="Run PCA analysis for multiple K values sequentially")
    ap.add_argument("--out_dir", type=str, required=True, help="Base output directory")
    args = ap.parse_args()

    # Resolve to absolute path before changing cwd for subprocesses
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Run subprocesses from the repo directory (where pca_analysis_of_qsp.py lives)
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    failed = []
    for K in K_VALUES:
        cmd = [
            sys.executable, "pca_analysis_of_qsp.py",
            "--K", str(K),
            "--out_dir", os.path.join(out_dir, f"pca_output_K={K}"),
            "--device", "auto",
            "--skip_tests",
        ]
        print(f"\n{'='*60}")
        print(f"Running K={K}: {' '.join(cmd)}")
        print(f"{'='*60}")
        rc = subprocess.call(cmd, cwd=repo_dir)
        if rc != 0:
            failed.append(K)
            print(f"K={K} FAILED (exit code {rc})")
        else:
            print(f"K={K} completed successfully")

    if failed:
        print(f"\nFailed K values: {failed}")
        sys.exit(1)
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()
