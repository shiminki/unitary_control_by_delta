# Detuning-Selective Unitary Control via QSP + Neural Network + PCA

A compact pipeline for designing detuning-selective single-qubit gates via
**Quantum Signal Processing (QSP)**.  A neural network maps four target
rotation angles `(α₀, α₁, α₂, α₃)` to a QSP phase vector `φ ∈ ℝ^{K+1}`, so
that the resulting pulse implements `R_x(αᵢ)` for detunings near the `i`-th
target centre, and PCA compresses the learned `φ(α)` trajectory into a small
number of analytically expressible components.

## System

| Parameter | Value |
|-----------|-------|
| Detuning centres | Δ ∈ {−100, −32, +32, +100} MHz |
| Max detuning Δ₀ | 200 MHz |
| Rabi frequency Ω | ∈ {20, 40, 80} MHz |
| Robustness window | ±10 MHz |
| QSP degree K | ∈ {50, 70, 100, 160} |

## Problem

Given `Ω`, `K`, and a target vector `(α₀..α₃)`, produce `φ = (φ₀..φ_K)` such that
the physical-basis sequence

```
R_x(φ₀; δ) · R_z(θ) · R_x(φ₁; δ) · R_z(θ) · … · R_x(φ_K; δ)  =  R_x(αᵢ)
```

holds whenever `|δ − δᵢ| < σ` for any of the four target peaks `i`.  Here
`R_x(φ; δ) = exp(−i/2 · (sign(φ)·Ω·X + δ·Z) · |φ|)` is the driven control
operator in the presence of detuning; `R_z(θ)` is the free-evolution signal
operator with `θ = π/2 · (1 + δ/Δ₀)`.

## Architecture

`PulseNet(Ω, K)` — an 8-layer MLP with 512 hidden units and SiLU activations.

- **Input**: `(B, 4)` rotation angles.
- **Encoding**: Fourier features `[cos(k·α/2), sin(k·α/2)]` for `k = 1..8`.  The
  half-angle matches the 4π fundamental period of `R_x(α)` in SU(2).
- **Output**: `(B, K+1)` QSP phases in float64.
- **Training data**: 65 536 IID samples of `α ∈ [−ε, 4π+ε]^4` with `ε = 0.01`.
- **Eval data**: 1024 held-out samples.
- **Batch / optim**: 512, Adam + CosineAnnealingLR, 40 epochs, gradient
  clipping 10.0, float64 throughout.
- **Loss**: vectorized `|u₀₀(pred) − exp(−iα/2)|²` averaged over 128 detunings
  per peak (robustness window), computed via `build_qsp_unitary_batched`.

## Repository layout

```
neural_network_optimization/
├── constants.py          # system constants, Ω / K grids
├── physics.py            # signal / control operators, QSP unitary, fidelity
├── model.py              # PulseNet, phi_to_pulse_df, get_runtime_from_phi
├── data.py               # uniform α sampler + fixed 65 536 / 1024 dataset
├── train.py              # training loop + CLI
├── inference.py          # load_model, generate_pulse/phi, compute_fidelity, compute_runtime
├── scaling_law.py        # grid over (Ω, K); writes CSVs + figures; triggers PCA
├── pca_analysis.py       # per-peak PCA on PulseNet phi(α); Fourier / poly fits
├── plotting.py           # matrix-element, scaling law, runtime-fidelity, PCA figures
├── tests/                # pytest unit tests (23 cases)
└── weights/              # joint_Omega{Ω}_K{K}.pt (12 files after full run)
outputs/                  # scaling_law_*.csv, figures, pca/Omega{Ω}_K{K}/peak{i}/*
streamlit_app.py          # demo: load-only, plus PCA tab
```

## Usage

Install:

```bash
pip install -r requirements.txt
```

### Unit tests (fast)

```bash
pytest neural_network_optimization/tests/
```

### Train a single (Ω, K)

```bash
python -m neural_network_optimization.train --Omega 80 --K 70
```

Weights land in `neural_network_optimization/weights/joint_Omega80.0_K70.pt`.

### Full scaling law (trains all 12 configs, writes summary + figures + PCA)

```bash
python -m neural_network_optimization.scaling_law
```

Outputs (under `outputs/`):
- `scaling_law_full.csv` (6 144 rows: one per trial).
- `scaling_law_summary.csv` (12 rows: per-(Ω, K) avg + min + std fidelity + mean runtime).
- `scaling_law.png` (avg + min fidelity vs K, one line per Ω).
- `runtime_vs_fidelity.png` (runtime vs fidelity / infidelity scatter).
- `matrix_element_Omega80_K70.png` (canonical matrix-element plot).
- `pca/Omega80_K70/peak{i}/` (overview + amplitudes + comparative pulses + comparative
  matrix elements PNGs, three coefficient CSVs, analytical-form text per peak).

Pass `--small` for a tiny test grid (`Ω=40, K ∈ {20, 30}`, 8 trials each).

### PCA for a single model

```bash
python -m neural_network_optimization.pca_analysis --Omega 80 --K 70
```

### Inference from Python

```python
from neural_network_optimization import load_model, generate_pulse, compute_fidelity

model = load_model(Omega=80, K=70)
pulse_df = generate_pulse(model, [math.pi/2, math.pi, math.pi/3, 0.0])
fidelity = compute_fidelity(model, [math.pi/2, math.pi, math.pi/3, 0.0])
```

### Interactive demo

```bash
streamlit run streamlit_app.py
```

Sidebar dropdowns pick `(Ω, K)`; sliders pick the four `αᵢ`.  Tabs show the
pulse schedule, the matrix-element vs detuning plot, the numerical fidelity
and runtime, the `φ` bar plot, and the per-peak PCA decomposition with
Fourier and polynomial fits.  The app never trains — if weights are missing
it prints the command to generate them.

## Reference

Gradient-based QSP: [arXiv:2312.08426](https://arxiv.org/abs/2312.08426).
