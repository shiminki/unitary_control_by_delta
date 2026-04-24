"""Neural-network QSP pulse generator + scaling law + PCA analysis."""

from .constants import (
    DELTA_CENTERS_MHZ,
    DELTA_0_MHZ,
    N_PEAKS,
    OMEGA_MHZ,
    ROBUSTNESS_WINDOW_MHZ,
    DEFAULT_K,
    EPS,
    ALPHA_RANGE,
    OMEGA_LIST,
    K_LIST,
)
from .model import PulseNet, phi_to_pulse_df, get_runtime_from_phi, get_weight_path
from .inference import (
    load_model,
    generate_phi,
    generate_pulse,
    compute_fidelity,
    compute_runtime,
)
from .train import train
