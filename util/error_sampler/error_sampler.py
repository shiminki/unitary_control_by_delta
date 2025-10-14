import torch
from typing import Dict


###############################################################################
# Error Samplers
################################################################################


def single_qubit_error_sampler(total_samples, device):
    """
    Generate static error samples for single qubit control.

    Parameters
    ----------
    total_samples : int
        Total number of error samples to generate.
    device : torch.device
        Device on which to create the error samples.

    Returns
    -------
    torch.Tensor
        Shape ``(total_samples, 2)`` – each row is ``[delta, epsilon]``
        (off-resonant detuning, pulse length error).
    """
    delta = (torch.rand(total_samples, device=device) * 2 - 1) * 1.5  # (-1.5, 1.5)
    epsilon = (torch.rand(total_samples, device=device) * 2 - 1) * 0.1  # (-0.1, 0.1)

    return torch.stack([delta, epsilon], dim=-1)  # (total_samples, 2)



def two_qubit_error_sampler(total_samples, device):
    """
    Generate static error samples for two qubit entangled control.

    Parameters
    ----------
    total_samples : int
        Total number of error samples to generate.
    device : torch.device
        Device on which to create the error samples.

    Returns
    -------
    torch.Tensor
        Shape ``(total_samples, 4)`` – each row is ``[delta_sys, epsilon, delta_anc, coupling_error]``
        (off-resonant detuning of system and ancila, pulse length error, coupling error).
    """
    delta_sys = (torch.rand(total_samples, device=device) * 2 - 1) * 1.5  # (-1.5, 1.5)
    delta_anc = (torch.rand(total_samples, device=device) * 2 - 1) * 1.5  # (-1.5, 1.5)
    epsilon = (torch.rand(total_samples, device=device) * 2 - 1) * 0.1  # (-0.1, 0.1)
    coupling_error = (torch.rand(total_samples, device=device) * 2 - 1) * 0.1  # (-0.1, 0.1)

    return torch.stack([delta_sys, epsilon, delta_anc, coupling_error], dim=-1)  # (total_samples, 4)

