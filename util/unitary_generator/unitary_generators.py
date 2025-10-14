import torch
from typing import Dict


###############################################################################
# Pauli matrices and helpers – cached per device
###############################################################################

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)

# Simple †‑immortal cache keyed by torch.device.
_PAULI_CACHE: Dict[torch.device, torch.Tensor] = {}
_I2_CACHE: Dict[torch.device, torch.Tensor] = {}
_I4_CACHE: Dict[torch.device, torch.Tensor] = {}


def _get_paulis(device: torch.device) -> torch.Tensor:
    """Return a stack ``(4, 2, 2)`` of *(I, σₓ, σ_y, σ_z)* on *device*.

    The tensors are created on their first use on each device and then reused
    to avoid needless kernel launches and host‑to‑device traffic.
    """
    if device not in _PAULI_CACHE:
        _PAULI_CACHE[device] = torch.stack(
            [_I2_CPU, _SIGMA_X_CPU, _SIGMA_Y_CPU, _SIGMA_Z_CPU], dim=0
        ).to(device)
    return _PAULI_CACHE[device]


def _get_I2(device: torch.device, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
    """Get 2x2 identity matrix on specified device."""
    key = (device, dtype)
    if key not in _I2_CACHE:
        _I2_CACHE[key] = torch.eye(2, dtype=dtype, device=device)
    return _I2_CACHE[key]


def _get_I4(device: torch.device, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
    """Get 4x4 identity matrix on specified device."""
    key = (device, dtype)
    if key not in _I4_CACHE:
        _I4_CACHE[key] = torch.eye(4, dtype=dtype, device=device)
    return _I4_CACHE[key]


_ZZ_CACHE = {}

def _zz(device, dtype):
    key = (device, dtype)
    if key not in _ZZ_CACHE:
        pauli = _get_paulis(device).type(dtype)
        _ZZ_CACHE[key] = torch.kron(pauli[3], pauli[3]).to(device).type(dtype).contiguous()
    return _ZZ_CACHE[key]


###############################################################################
# Unitary generators
###############################################################################


def batched_unitary_generator_single_qubit(
        pulses: torch.Tensor,
        error: torch.Tensor,
    ) -> torch.Tensor:
        """Compose the total unitary for a **batch** of composite sequences.

        Parameters
        ----------
        pulses : torch.Tensor
            Shape ``(B, L, 3)``, where each pulse is
            ``[Ω, φ, t]`` (detuning, Rabi amplitude, phase, duration).
        error : torch.Tensor
            Shape ``(B, 2)`` static off‑resonant detuning and pulse length error for each
            batch element.  If you fuse Monte‑Carlo repeats into the batch, just
            expand ``delta`` accordingly.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 2, 2)`` complex64/128 – the composite unitary ``U_L ⋯ U_1``.
        """

        if pulses.ndim != 3 or pulses.shape[-1] != 3:
            raise ValueError("'pulses' must have shape (B, L, 3)")
        
        if error.ndim != 2 or error.shape[1] != 2:
            raise ValueError("'error' must have shape (B, 2)")


        B, L, _ = pulses.shape
        device = pulses.device
        dtype = torch.cdouble

        # Unpack and reshape to broadcast with Pauli matrices.
        omega, phi, tau = pulses.unbind(dim=-1)  # each (B, L)

        # (4, 2, 2) on correct device
        pauli = _get_paulis(device).type(dtype)

        # ORE and PLE
        delta = error[:, 0]
        epsilon = error[:, 1]

        # Build base Hamiltonian H₀ for every pulse in parallel.
        H_base = omega[..., None, None] / 2 * (
            torch.cos(phi)[..., None, None] * pauli[1]
            + torch.sin(phi)[..., None, None] * pauli[2]
        )
        
        H = H_base + delta[..., None, None, None] * pauli[3]

        H = 0.5 * H * (1 + epsilon[..., None, None, None])

        # U_k = exp(-i H_k t_k)
        U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 2, 2)


        # U: (B, L, 2, 2)   want: U[:, L-1] @ ... @ U[:, 1] @ U[:, 0]
        X = U
        I = torch.eye(2, dtype=dtype, device=device).expand(B, 1, 2, 2)

        while X.size(1) > 1:
            # pad to even length
            if (X.size(1) & 1) == 1:
                X = torch.cat([X, I], dim=1)
            # pairwise multiply preserving left-to-right order:
            # (U1 @ U0), (U3 @ U2), ...
            X = X[:, 1::2] @ X[:, 0::2]

        U_out = X[:, 0]  # (B, 2, 2)


        return U_out


def batched_unitary_generator_two_qubits_entangled(
    pulses: torch.Tensor,
    error: torch.Tensor,
    J: float = 0.1,
) -> torch.Tensor:
    """Compose the total unitary for a **batch** of composite sequences.

    Parameters
    ----------
    pulses : torch.Tensor
        Shape ``(B, L, 5)``, where each pulse is
        ``[Ω_sys, φ_sys, Ω_anc, φ_anc, t]`` (detuning, Rabi amplitude, phase, duration).
    error : torch.Tensor
        Shape ``(B, 4)`` static off‑resonant detuning of system and ancila, and pulse length error for each
        batch element.  If you fuse Monte‑Carlo repeats into the batch, just
        expand ``delta`` accordingly.

    Returns
    -------
    torch.Tensor
        Shape ``(B, 4, 4)`` complex64/128 – the composite unitary ``U_L ⋯ U_1``.
    """

    if pulses.ndim != 3 or pulses.shape[-1] != 5:
        raise ValueError(f"'pulses' must have shape (B, L, 5). Input pusle has shape {pulses.shape}")
    
    if error.ndim != 2 or error.shape[1] != 4:
        raise ValueError("'error' must have shape (B, 4)")

    B, L, _ = pulses.shape
    device = pulses.device
    dtype = torch.cfloat

    # Unpack and reshape to broadcast with Pauli matrices.
    phi_sys, omega_sys, phi_anc, omega_anc, tau = pulses.unbind(dim=-1)  # each (B, L)

    # (4, 2, 2) on correct device
    pauli = _get_paulis(device).type(dtype)
    I2 = _get_I2(device, dtype)  # Use cached identity on correct device
    I4 = _get_I4(device, dtype)  # Use cached 4x4 identity

    # ORE and PLE
    delta_sys = error[:, 0]
    epsilon = error[:, 1]
    delta_anc = error[:, 2]
    coupling_error = error[:, 3]

    def build_single_hamiltonian(omega, phi, delta, epsilon):
        H_base = omega[..., None, None] * (
            torch.cos(phi)[..., None, None] * pauli[1]
            + torch.sin(phi)[..., None, None] * pauli[2]
        )
        H = H_base + delta[..., None, None, None] * pauli[3]
        H = 0.5 * H * (1 + epsilon[..., None, None, None])
        return H

    # Build base Hamiltonian H₀ for every pulse in parallel.
    H_sys = build_single_hamiltonian(omega_sys, phi_sys, delta_sys, epsilon)
    H_anc = build_single_hamiltonian(omega_anc, phi_anc, delta_anc, epsilon)

    # Build interaction Hamiltonian
    H_int = 0.5 * (1 + coupling_error[..., None, None, None]) * (J * _zz(device, dtype))  # (4, 4)

    # Construct full Hamiltonian using kronecker products
    # H = H_sys ⊗ I + I ⊗ H_anc + H_int
    # Shape: (B, L, 4, 4)
    
    # Expand dimensions for kronecker product
    H_sys_kron = torch.zeros(B, L, 4, 4, dtype=dtype, device=device)
    H_anc_kron = torch.zeros(B, L, 4, 4, dtype=dtype, device=device)
    
    for b in range(B):
        for l in range(L):
            H_sys_kron[b, l] = torch.kron(H_sys[b, l], I2)
            H_anc_kron[b, l] = torch.kron(I2, H_anc[b, l])
    
    H = H_sys_kron + H_anc_kron + H_int

    # U_k = exp(-i H_k t_k)
    U = torch.linalg.matrix_exp(-1j * H * tau[..., None, None])  # (B, L, 4, 4)

    # Compose unitaries: U[:, L-1] @ ... @ U[:, 1] @ U[:, 0]
    # Use the efficient pairwise multiplication approach
    X = U
    I4_expanded = I4.expand(B, 1, 4, 4)

    while X.size(1) > 1:
        # pad to even length
        if (X.size(1) & 1) == 1:
            X = torch.cat([X, I4_expanded], dim=1)
        # pairwise multiply preserving left-to-right order
        X = X[:, 1::2] @ X[:, 0::2]

    U_out = X[:, 0]  # (B, 4, 4)

    return U_out