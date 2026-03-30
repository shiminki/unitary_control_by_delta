"""Run PCA analysis for multiple K values in parallel.

Usage (Colab A100):
    python run_all_K.py

Each K value runs as a separate subprocess, all sharing the GPU.
"""

import argparse
import os
import subprocess
import sys

K_VALUES = [100, 150, 200, 300]


def main():
    ap = argparse.ArgumentParser(description="Run PCA analysis for multiple K values in parallel")
    ap.add_argument("--out_dir", type=str, required=True, help="Base output directory")
    args = ap.parse_args()

    procs = []
    for K in K_VALUES:
        cmd = [
            sys.executable, "pca_analysis_of_qsp.py",
            "--K", str(K),
            "--out_dir", os.path.join(args.out_dir, f"pca_output_K={K}"),
            "--device", "auto",
            "--skip_tests",
        ]
        print(f"Launching K={K}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        procs.append((K, proc))

    # Wait for all to finish
    failed = []
    for K, proc in procs:
        rc = proc.wait()
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
