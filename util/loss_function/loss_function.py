import torch


def negative_log_loss(U_out, U_target, fidelity_fn=None, num_qubits=1):
    """Negative log loss function."""
    if fidelity_fn is None:
        raise ValueError("fidelity_fn must be provided")
    return -torch.log(torch.mean(fidelity_fn(U_out, U_target, num_qubits)))


def infidelity_loss(U_out, U_target, fidelity_fn=None, num_qubits=1):
    """Infidelity loss function."""
    if fidelity_fn is None:
        raise ValueError("fidelity_fn must be provided")
    return 1 - torch.mean(fidelity_fn(U_out, U_target, num_qubits))


def sharp_loss(U_out, U_target, fidelity_fn=None, num_qubits=1, tau=0.99, k=100):
    """Sharp loss function with configurable parameters."""
    if fidelity_fn is None:
        raise ValueError("fidelity_fn must be provided")
    
    F = torch.mean(fidelity_fn(U_out, U_target, num_qubits))
    return custom_loss(F, tau, k)


def custom_loss(x, tau=0.99, k=100):
    """Custom loss function for fidelity optimization."""
    return torch.log(1 + torch.exp(-k * (x - tau))) * (1 - x)