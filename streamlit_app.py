import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

from qsp_fit_relaxed import TrainConfig, train, plot_matrix_element_vs_delta


def parse_float_list(raw: str) -> Tuple[List[float], str]:
    cleaned = raw.replace("\n", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return [], "Enter a comma-separated list of numbers."
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            return [], f"Could not parse '{p}' as a number."
    return values, ""


st.set_page_config(page_title="Controlled Unitary Demo", layout="wide")
st.title("Controlled Unitary Demo via QSP with Relaxed Constraints")


description = r"""
This demo allows you to generate control sequence for implementing target X-rotations
on a qubit using detuning as the control parameter. For a given list of delta_vals
and alpha_vals, we train a QSP sequence of phases to realize $R_x(\alpha_i)$ when detuning is near
$\delta_i$. Specifically, the target is to achieve

$R_x(\alpha_i)$ when $\delta = \delta_i \pm \sigma$.

for some window width $\sigma$ specified by `signal_window` = $\sigma$.

 Key parameters are 

`N`: Number of target gates (or number of peaks in the detuning distribution)

`K`: NUmber of control gates. QSP sequence will generate polynomial $P(x)$ of degree at most $K$.

`Delta_0`: Maximum detuning (2$\pi$ MHz). Detuning distributions should be within $[-\Delta_0, \Delta_0]$.

`delta_vals`: List of detuning values (2$\pi$ MHz) at which target gates are defined.

`alpha_vals`: List of target rotation angles (in units of pi) corresponding to each delta

`signal_window`: Width of the detuning signal window (2$\pi$ MHz).

`build_with_detuning`: Whether to include detuning in the signal operator of QSP. When set to False, we are assuming infinitely fast control.
"""

st.markdown(description)

st.subheader("Disclaimer and Setup Instructions")

disclaimer = """
With `build_with_detuning` enabled, a reasonable `K` should be around 100. 
However, the streamlit server will take a while to run (~30 min), and we recommend to run this demo locally (~10 min). 
To do so, please follow the instruction below:

1. Clone the repository: `git clone https://github.com/shiminki/unitary_control_by_delta.git`
2. Install the necessary requirements: `pip install -r requirements.txt`
3. Run the streamlit app: `streamlit run streamlit_app.py`
"""

st.markdown(disclaimer)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core Arguments")
    K = st.number_input("K (max phase index)", min_value=1, value=30, step=1)
    N = st.number_input("N (num_peaks = number of gates)", min_value=1, value=4, step=1)
    Delta_0 = st.number_input("Delta_0 (MHz)", min_value=0.0, value=100.0, step=1.0)
    signal_window = st.number_input("signal_window (MHz)", min_value=0.0, value=5.0, step=0.1)
    Omega_max = st.number_input("Omega_max (MHz)", min_value=0.0, value=80.0, step=1.0)
    build_with_detuning = st.checkbox("build_with_detuning", value=False)
    

with col2:
    st.subheader("Other Arguments")
    steps = st.number_input("training steps", min_value=1, value=2000, step=100)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = st.number_input("lr", min_value=0.0, value=5e-2, step=1e-3, format="%.6f")
    device = st.selectbox("device", options=["cpu", "cuda"], index=0 if default_device == "cpu" else 1)
    end_with_W = st.checkbox("end_with_W", value=False)
    out_dir = st.text_input("out_dir", value="plots_relaxed")
    

st.subheader("Target Values")
alpha_default = "0.3333, 1, 0.5, 1.25"
delta_default = "-80, -32, 32, 80"
alpha_raw = st.text_area("alpha_vals (radians unit in pi, length N)", value=alpha_default, height=80)
delta_raw = st.text_area("delta_vals (MHz, length N)", value=delta_default, height=80)

run_btn = st.button("Run Training")

if run_btn:
    alpha_list, alpha_err = parse_float_list(alpha_raw)
    delta_list, delta_err = parse_float_list(delta_raw)

    errors = []
    if alpha_err:
        errors.append(f"alpha_vals: {alpha_err}")
    if delta_err:
        errors.append(f"delta_vals: {delta_err}")
    if not errors:
        if len(alpha_list) != N:
            errors.append(f"alpha_vals length is {len(alpha_list)}, expected N={N}.")
        if len(delta_list) != N:
            errors.append(f"delta_vals length is {len(delta_list)}, expected N={N}.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(42)
        torch.set_default_dtype(torch.float64)

        if device == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA not available; falling back to CPU.")
            device = "cpu"

        cfg = TrainConfig(
            Omega_max=2 * math.pi * Omega_max,
            Delta_0=2 * math.pi * Delta_0,
            singal_window=2 * math.pi * signal_window,
            K=int(K),
            steps=int(steps),
            lr=float(lr),
            device=device,
            end_with_W=end_with_W,
            out_dir=out_dir,
            build_with_detuning=build_with_detuning,
        )

        delta_vals = torch.tensor(delta_list, device=device) * (2 * math.pi)
        alpha_vals = torch.tensor(alpha_list, device=device) * math.pi

        if (delta_vals.abs() > cfg.Delta_0).any():
            st.warning("Some delta_vals exceed |Delta_0| after unit conversion.")

        with st.spinner("Training..."):

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def progress_cb(step: int, total: int, loss: float, eta: float) -> None:
                progress_bar.progress(step / total)
                eta_min = int(eta // 60)
                eta_sec = int(eta % 60)
                progress_text.write(
                    f"Step {step}/{total} — loss {loss:.3e} — ETA {eta_min:02d}:{eta_sec:02d}"
                )

            phi_final, final_loss = train(
                cfg,
                delta_vals,
                alpha_vals,
                sample_size=2048,
                progress_cb=progress_cb
            )

        st.success(f"Training complete. Final loss: {final_loss:.6e}")

        phi_df = pd.DataFrame({"index": np.arange(len(phi_final)), "phi": phi_final.numpy()})
        csv_path = os.path.join(
            out_dir,
            f"learned_phases_K={int(K)}_final_loss_{final_loss:.6f}noisy_control_{cfg.build_with_detuning}.csv",
        )
        phi_df.to_csv(csv_path, index=True)

        st.write(f"Saved phases CSV: `{csv_path}`")
        st.download_button(
            label="Download phases CSV",
            data=phi_df.to_csv(index=True).encode("utf-8"),
            file_name=os.path.basename(csv_path),
            mime="text/csv",
        )
        plot_path = os.path.join(out_dir, f"u00_final_K={int(K)}.png")

        plot_matrix_element_vs_delta(
            phi_final, cfg.Omega_max, delta_vals, alpha_vals,
            cfg.Delta_0, cfg.end_with_W, cfg.device,
            out_path=plot_path,
            delta_width=cfg.singal_window,
            build_with_detuning=cfg.build_with_detuning,
        )

        if os.path.exists(plot_path):
            st.subheader("Matrix Element vs Delta Plot")
            st.image(plot_path, caption="Matrix element vs delta", use_container_width=True)
        else:
            st.info(f"Plot not found at `{plot_path}`")

        st.write("Args used:", {
            "K": int(K),
            "steps": int(steps),
            "num_peaks": int(N),
            "Delta_0": float(Delta_0),
            "signal_window": float(signal_window),
            "Omega_max": float(Omega_max),
            "lr": float(lr),
            "device": device,
            "end_with_W": end_with_W,
            "out_dir": out_dir,
            "build_with_detuning": build_with_detuning,
        })
