# Unitary Control by Detuning (QSP)

Gradient-based Quantum Signal Processing (QSP) for implementing target
single-qubit X-rotations using detuning as the control parameter.
Given a list of detuning peaks δᵢ and target angles αᵢ, the framework learns
a phase sequence φ ∈ ℝ^{K+1} such that

> `build_qsp_unitary(φ, δ) ≈ Rₓ(αᵢ)`  for all  `δ ∈ [δᵢ − σ, δᵢ + σ]`

Two complementary approaches are provided:

| Approach | File / Package | Description |
|----------|---------------|-------------|
| **Classical (per-instance)** | `single_pulse_optimization_QSP/` | Adam-based optimisation of φ for a fixed (δᵢ, αᵢ) set |
| **Neural network (amortised)** | `neural_network_optimization_QSP/` | MLP trained once over random α configs; predicts φ instantly at inference |

---

## Repository structure

```
.
├── single_pulse_optimization_QSP/
│   ├── qsp_fit_x_rotation.py   # core QSP physics + classical trainer
│   └── __init__.py
│
├── neural_network_optimization_QSP/
│   ├── qsp_phase_net.py        # QSPPhaseNet model + NNTrainConfig + train_nn
│   └── __init__.py
│
├── streamlit_app.py            # interactive demo (both approaches)
├── scaling_law.py              # grid sweep over Omega x K (classical + NN)
├── util.py                     # fidelity helpers, Bloch animation, pulse plots
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/shiminki/unitary_control_by_delta.git
cd unitary_control_by_delta
pip install -r requirements.txt
```

For Bloch-sphere animations (optional):

```bash
pip install qutip
```

---

## Key parameters

| Symbol | CLI name | Meaning |
|--------|----------|---------|
| K | `--K` | QSP order; phase vector φ has length K+1 |
| Ω_max | `--Omega_max` | Maximum Rabi frequency (MHz) |
| Δ₀ | `--Delta_0` | Maximum detuning range (MHz); δ ∈ [−Δ₀, Δ₀] |
| σ | `--robustness_window` | Half-width of the robustness window (MHz) |
| N | `--N` / `--num_peaks` | Number of detuning peaks / target gates |
| δᵢ | `--delta_vals` | Peak detuning locations (MHz) |
| αᵢ | alpha_vals input | Target rotation angles (units of π) |

All internal computations use **angular units** (rad/μs).  
Convert: `value_rad = 2π × value_MHz`.

---

## 1 · Streamlit demo

```bash
streamlit run streamlit_app.py
```

The app has two sections:

### Classical QSP optimisation
Configure K, Ω_max, Δ₀, σ, δ-peaks, α-targets and click **Run Training**.
Results tabs show the pulse schedule, matrix element vs δ, fidelity contour,
fidelity-vs-std, and an optional Bloch-sphere animation.

### Neural Network QSP
Configure NN training steps, batch size, and optionally a `peak_index`
(to train with only one α varying). Click **Train Neural Network**,
then use the prediction panel to get an instant φ prediction and fidelity
plot for any α configuration.

---

## 2 · Classical optimisation CLI

```bash
python -m single_pulse_optimization_QSP.qsp_fit_x_rotation \
    --K 70 \
    --num_peaks 4 \
    --Omega_max 80 \
    --Delta_0 200 \
    --robustness_window 10 \
    --steps 8000 \
    --out_dir plots_relaxed
```

This trains a single φ for the default 4-peak configuration and saves a
matrix-element plot and the learned phases to `--out_dir`.

---

## 3 · Neural network training CLI

```bash
python -m neural_network_optimization_QSP.qsp_phase_net \
    --K 70 --N 4 \
    --Omega_max 80 --Delta_0 200 --robustness_window 10 \
    --delta_vals -100 -32 32 100 \
    --steps 10000 --batch_size 64 --lr 1e-3 \
    --out_dir nn_qsp_output
```

**Single-peak mode** — train with only α at peak index 0 varying:

```bash
python -m neural_network_optimization_QSP.qsp_phase_net \
    --K 70 --N 4 --peak_index 0 \
    --delta_vals -100 -32 32 100 \
    --steps 10000 --out_dir nn_qsp_peak0
```

**Programmatic usage:**

```python
import math, torch
from neural_network_optimization_QSP import NNTrainConfig, train_nn, predict_phi

cfg = NNTrainConfig(
    K=70, N=4,
    Omega_max=2*math.pi*80,
    Delta_0=2*math.pi*200,
    robustness_window=2*math.pi*10,
    delta_vals=[2*math.pi*d for d in [-100, -32, 32, 100]],
    steps=10_000,
    batch_size=64,
    peak_index=None,   # None = all peaks vary; int i = only peak i varies
)
model, train_losses, eval_records = train_nn(cfg)

alpha_query = torch.tensor([0.5*math.pi, math.pi, 0.3*math.pi, 1.5*math.pi])
phi = predict_phi(model, alpha_query)   # shape (K+1,) — instant, no optimisation
```

Outputs saved to `--out_dir`:
- `model_final.pt` — best checkpoint (by held-out eval loss)
- `training_curve.png` — train + eval loss vs step
- `ckpt_step*.pt` — intermediate checkpoints

---

## 4 · Scaling law sweep

Sweeps the grid **Ω_max ∈ {40, 80, 120, 160} MHz × K ∈ {50, 70, 100}**.

```bash
# Both classical (30 trials/config) and NN (one model/config)
python scaling_law.py --out_dir scaling_law_results

# NN only
python scaling_law.py --skip_classical true --out_dir scaling_law_results

# Classical only
python scaling_law.py --skip_nn true --out_dir scaling_law_results

# Quick smoke-test (small grid, 2 trials, 200 NN steps)
python scaling_law.py --small true --skip_classical true
```

Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--nn_steps` | 5000 | Training steps per NN model |
| `--nn_batch_size` | 64 | Batch size for NN training |
| `--num_trials` | 30 | Classical trials per (Ω, K) config |
| `--max_workers` | 6 | Parallel workers for classical trials |
| `--skip_classical` | false | Skip classical sweep |
| `--skip_nn` | false | Skip NN sweep |
| `--small` | false | Reduced grid for quick testing |

Outputs:
- `scaling_law_results/scaling_law_classical.csv` — per-trial fidelity & runtime
- `scaling_law_results/scaling_law_nn.csv` — per-(Ω, K) NN eval loss & avg fidelity
- `scaling_law_results/nn_data/Omega{X}_K{Y}/` — per-model checkpoints, training curves, per-config fidelity CSV

---

## Physics background

The QSP sequence alternates **control pulses** and **signal operators**:

```
U = R_z(φ_0; δ) · W(θ) · R_z(φ_1; δ) · W(θ) · … · R_z(φ_K; δ)
```

- **Signal operator** `W(θ) = Rₓ(θ)` with `θ = π/2 · (1 + δ/Δ₀)`,
  realised by waiting `τ = π/(2Δ₀)` with Ω = 0.
- **Control operator** `R_z(φ; δ)` — drive at Rabi frequency Ω_max for time
  `|φ|/Ω_max`, with detuning δ leaking in as an off-axis tilt.

In the physical (lab) basis the sequence implements `Rₓ(αᵢ)` near δ = δᵢ.
The **neural network** encodes each target gate as `[cos(αᵢ/2), sin(αᵢ/2)]`
and learns the mapping to φ across all α configurations in a single training run.
