import torch


def negative_log_loss(U_out, U_target, fidelity_fn, num_qubits):
    return -torch.log(torch.mean(fidelity_fn(U_out, U_target, num_qubits)))


def infidelity_loss(U_out, U_target, fidelity_fn, num_qubits):
    return 1 - torch.mean(fidelity_fn(U_out, U_target, num_qubits))


def sharp_loss(U_out, U_target, fidelity_fn=None, num_qubits=1, tau=0.99, k=100):
    assert fidelity_fn is not None, "fidelity_fn must be provided"

    F = torch.mean(fidelity_fn(U_out, U_target, num_qubits))
    return custom_loss(F, tau, k)

def custom_loss(x, tau=0.99, k=100):
    return torch.log(1 + torch.exp(-k * (x - tau))) * (1 - x)

