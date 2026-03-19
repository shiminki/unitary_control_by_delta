"""PulseGeneratorNet: neural network mapping theta -> QSP phases."""

import math

import torch
import torch.nn as nn

from .constants import DEFAULT_K, DELTA_CENTERS_MHZ, OMEGA_MHZ, DELTA_0_MHZ, ROBUSTNESS_WINDOW_MHZ


class PulseGeneratorNet(nn.Module):
    """Neural network mapping rotation angle theta -> QSP phases phi[0..K].

    Parameters
    ----------
    peak_index : int
        Which detuning peak (0..N-1) this network targets.
    K : int
        QSP sequence degree (phi has K+1 elements).
    hidden_dim : int
        Width of hidden layers.
    num_layers : int
        Number of hidden layers.
    n_freq : int
        Number of Fourier frequencies for the input encoding.
        Input dim = 2 * n_freq: [cos(theta), sin(theta), cos(2*theta), ...].
    delta_centers_mhz : list
        Detuning peak centers in MHz.
    Omega_mhz : float
        Rabi frequency in MHz.
    Delta_0_mhz : float
        Maximum detuning range in MHz.
    robustness_window_mhz : float
        Half-width of robustness window in MHz.
    """

    def __init__(
        self,
        peak_index: int,
        K: int = DEFAULT_K,
        hidden_dim: int = 128,
        num_layers: int = 4,
        n_freq: int = 8,
        delta_centers_mhz: list = None,
        Omega_mhz: float = OMEGA_MHZ,
        Delta_0_mhz: float = DELTA_0_MHZ,
        robustness_window_mhz: float = ROBUSTNESS_WINDOW_MHZ,
    ):
        super().__init__()
        if delta_centers_mhz is None:
            delta_centers_mhz = list(DELTA_CENTERS_MHZ)

        self.peak_index = peak_index
        self.K = K
        self.n_freq = n_freq
        self.Omega_mhz = Omega_mhz
        self.Delta_0_mhz = Delta_0_mhz
        self.robustness_window_mhz = robustness_window_mhz
        self.delta_centers_mhz = delta_centers_mhz

        # Angular units for internal computation
        self.Omega_ang = 2 * math.pi * Omega_mhz
        self.Delta_0_ang = 2 * math.pi * Delta_0_mhz
        self.robustness_window_ang = 2 * math.pi * robustness_window_mhz
        self.delta_centers_ang = [2 * math.pi * d for d in delta_centers_mhz]

        # Fourier input encoding: [cos(k*theta), sin(k*theta)] for k=1..n_freq
        in_dim = 2 * n_freq
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, dtype=torch.float64))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, K + 1, dtype=torch.float64))
        self.mlp = nn.Sequential(*layers)

        # Initialize output layer small so initial phi ~ 0 (identity)
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)
            self.mlp[-1].bias.zero_()

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Map rotation angle(s) to QSP phase vectors.

        Parameters
        ----------
        theta : (B,) tensor of rotation angles in radians.

        Returns
        -------
        (B, K+1) tensor of QSP phase values.
        """
        # Fourier features with half-angle: [cos(θ/2), sin(θ/2), cos(θ), sin(θ), ...]
        # Using θ/2 gives a 4π-periodic encoding, matching the natural period of
        # R_x(θ). The naive θ encoding (period 2π) aliases θ=0 and θ=2π to the
        # same feature vector despite having opposite targets (u₀₀=+1 vs -1),
        # creating a gradient conflict that degrades fidelity near both endpoints.
        ks = torch.arange(1, self.n_freq + 1, dtype=theta.dtype, device=theta.device)
        # theta: (B,) -> (B, 1); ks: (n_freq,) -> angles: (B, n_freq)
        angles = (theta / 2).unsqueeze(-1) * ks.unsqueeze(0)
        x = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, 2*n_freq)
        return self.mlp(x)  # (B, K+1)
