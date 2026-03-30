"""QSP PCA analysis pipeline.

Modules
-------
constants : Physical constants
model : PulseGeneratorNet (theta -> QSP phases)
pulse : phi_to_pulse_df (phases -> pulse schedule)
training : train_nn, load_or_train_nn
evaluation : evaluate_nn, compute_fidelity_for_phi, verify_phi_fidelities
pca : pca_analysis, reconstruct_phi_analytical, reconstruct_phi_polyfit, format_analytical_form
plotting : plot_pca_overview, plot_pca_components, plot_amplitude_components, plot_tsne, plot_comparative_pulses, plot_comparative_matrix_elements
tests : run_tests, run_acceptance_test
"""

from .constants import (
    DELTA_CENTERS_MHZ, N_PEAKS, OMEGA_MHZ, DELTA_0_MHZ,
    ROBUSTNESS_WINDOW_MHZ, DEFAULT_K, EPS,
)
from .model import PulseGeneratorNet
from .pulse import phi_to_pulse_df
from .training import (
    presample_detunings, compute_batch_loss, train_nn, load_or_train_nn,
)
from .evaluation import (
    evaluate_nn, compute_fidelity_for_phi, verify_phi_fidelities,
)
from .pca import (
    pca_analysis, save_pca_csv, reconstruct_phi_analytical,
    reconstruct_phi_polyfit, format_analytical_form,
)
from .plotting import (
    plot_pca_overview, plot_pca_components, plot_amplitude_components,
    plot_tsne, plot_comparative_pulses, plot_comparative_matrix_elements,
)
from .tests import run_tests, run_acceptance_test
