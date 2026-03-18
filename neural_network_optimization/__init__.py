"""Neural network optimization pipeline for multi-peak QSP pulse generation.

Maps alpha_vals (N rotation angles) -> QSP phases phi[0..K] jointly,
so that the resulting pulse applies Rx(alpha_i) at each detuning peak i.
"""

from neural_network_optimization.model import (
    JointPulseGeneratorNet,
    phi_to_pulse_df,
    DELTA_CENTERS_MHZ,
    N_PEAKS,
    DEFAULT_K,
    DEFAULT_OMEGA_MHZ,
)
from neural_network_optimization.inference import (
    load_model,
    generate_pulse,
    compute_fidelity,
)
