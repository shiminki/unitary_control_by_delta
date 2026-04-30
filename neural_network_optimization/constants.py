"""Project-wide constants for the QSP pulse generator.

All frequency / detuning values are in nominal MHz.  Convert to angular units
(rad / us) by multiplying by 2*pi.
"""

import math

DELTA_CENTERS_MHZ = [-100.0, -32.0, 32.0, 100.0]
N_PEAKS = len(DELTA_CENTERS_MHZ)

DELTA_0_MHZ = 200.0
OMEGA_MHZ = 80.0
ROBUSTNESS_WINDOW_MHZ = 10.0

DEFAULT_K = 70

EPS = 0.01
ALPHA_RANGE = (-EPS, 4 * math.pi + EPS)

OMEGA_LIST = [40, 80, 120]
K_LIST = [50, 70, 100, 160]
