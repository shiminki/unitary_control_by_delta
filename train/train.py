"""
Problem Formulation:

Given a target range, unitary pair in the form

    {delta_j^start, delta_j^end, U_j}, j = 1, ..., N
    1 <= N <= 4

the goal is to find a phase control pulse phi(t) that implements the target unitary U_j
when detuning delta is in between delta_j^start and delta_j^end.

Evaluation:

We will uniformly draw detuning values uniformly from the union of all target detuning ranges,
and evaluate the average gate fidelity between the target unitary and the unitary implemented

F = 1/N sum_{j=1}^N int_{delta_j^start}^{delta_j^end} F(U_j, U_out(delta; phi(t))) d delta
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Iterator

import torch

from collections import defaultdict

import json
import argparse

import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from model.unitary_control_transformer import UnitaryControlTransformer
from model.unitary_control_trainer import UniversalModelTrainer


###############################################################################
# Pauli matrices and helpers – cached per device
###############################################################################

_I2_CPU = torch.eye(2, dtype=torch.cfloat)
_SIGMA_X_CPU = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cfloat)
_SIGMA_Y_CPU = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cfloat)
_SIGMA_Z_CPU = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)

# Simple †‑immortal cache keyed by torch.device.
_PAULI_CACHE: Dict[torch.device, torch.Tensor] = {}


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


###############################################################################
# Helper - unitary generation
###############################################################################


def batched_unitary_generator(
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


###############################################################################
# Loss and fidelity functions
###############################################################################


def fidelity(U_out: torch.Tensor, U_target: torch.Tensor, num_qubits: int=1) -> torch.Tensor:
    """Entanglement fidelity F = (|Tr(U_out^† U_target)|² + d) / d(d + 1)."""
    # trace over last two dims, keep batch
    # Batched conjugate transpose and matrix multiplication
    U_out_dagger = U_out.conj().transpose(-1, -2)  # [batch, 2, 2]
    product = U_out_dagger @ U_target  # [batch, 2, 2]

    # print(product, product.shape)

    # Batched trace calculation
    trace = torch.einsum('bii->b', product)  # [batch]
    trace_squared = torch.abs(trace) ** 2

    d = 2 ** num_qubits

    return (trace_squared + d) / (d * (d + 1))

def negative_log_loss(U_out, U_target, fidelity_fn, num_qubits):
    return -torch.log(torch.mean(fidelity_fn(U_out, U_target, num_qubits)))


def infidelity_loss(U_out, U_target, fidelity_fn, num_qubits):
    return 1 - torch.mean(fidelity_fn(U_out, U_target, num_qubits))


def sharp_loss(U_out, U_target, fidelity_fn=None, num_qubits=1, tau=0.99, k=100):
    if fidelity_fn is None:
        fidelity_fn = fidelity
    F = torch.mean(fidelity_fn(U_out, U_target, num_qubits))
    return custom_loss(F, tau, k)

def custom_loss(x, tau=0.99, k=100):
    return torch.log(1 + torch.exp(-k * (x - tau))) * (1 - x)




###############################################################################
# data
###############################################################################





def unit_vec(phi: float) -> tuple:
    """Unit vector (cos φ, sin φ, 0) lying in the x–y plane."""
    return (math.cos(phi), math.sin(phi), 0.0)


def build_SU2_dataset(dataset_size=10000, max_N=4, max_delta=1.5) -> List[torch.Tensor]:
    """Generate a batch of random SU(2) rotation vectors of various length."""

    assert dataset_size % max_N == 0, "dataset_size must be divisible by max_N"

    dataset = []

    for n in range(1, max_N + 1):
        for _ in range(dataset_size // max_N):
            # draw delta values
            deltas = torch.rand(2 * n) * 2 * max_delta - max_delta  # (-max_delta, max_delta)
            deltas = deltas.sort().values  # ascending order

            # draw unitaries
            data = []
            for i in range(n):
                phi = torch.rand(1).item() * 2 * math.pi
                n_x, n_y, n_z = unit_vec(phi)
                alpha = torch.rand(1).item() * 2 * math.pi  # rotation angle
                data.append(
                    torch.tensor(
                        [deltas[2 * i].item(), deltas[2 * i + 1].item(), n_x, n_y, n_z, alpha],
                        dtype=torch.float,
                    )
                )

            data_tensor = torch.stack(data, dim=0)  # (n, 6)
            dataset.append(data_tensor)

    return dataset


def create_dataloader(dataset: List[torch.Tensor], batch_size: int, shuffle: bool = True) -> Iterator[torch.Tensor]:
    """
    Create a dataloader that batches data with matching dimensions.
    
    Args:
        dataset: List of tensors with shape (n, 6) where n can vary
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle data within each length group
    
    Yields:
        Batched tensors of shape (batch_size, n, 6) or (remaining, n, 6) for trailing data
    """
    # Group dataset by sequence length n
    length_groups = defaultdict(list)
    
    for idx, data in enumerate(dataset):
        n = data.shape[0]  # sequence length
        length_groups[n].append(data)
    
    # Process each length group
    all_batches = []
    
    for n in sorted(length_groups.keys()):
        group_data = length_groups[n]
        
        # Shuffle within each group if requested
        if shuffle:
            indices = torch.randperm(len(group_data))
            group_data = [group_data[i] for i in indices]
        
        # Create batches for this length group
        num_full_batches = len(group_data) // batch_size
        
        # Full batches
        for i in range(num_full_batches):
            batch_list = group_data[i * batch_size : (i + 1) * batch_size]
            batch_tensor = torch.stack(batch_list, dim=0)  # (batch_size, n, 6)
            all_batches.append(batch_tensor)
        
        # Trailing batch (if any remaining data)
        remaining = len(group_data) % batch_size
        if remaining > 0:
            batch_list = group_data[num_full_batches * batch_size:]
            batch_tensor = torch.stack(batch_list, dim=0)  # (remaining, n, 6)
            all_batches.append(batch_tensor)
    
    # Optionally shuffle the order of batches across different lengths
    if shuffle:
        batch_indices = torch.randperm(len(all_batches))
        all_batches = [all_batches[i] for i in batch_indices]
    
    # Yield batches
    for batch in all_batches:
        yield batch


class SU2DataLoader:
    """
    PyTorch-style DataLoader for SU(2) dataset with length-aware batching.
    """
    
    def __init__(self, dataset: List[torch.Tensor], batch_size: int, shuffle: bool = True):
        """
        Args:
            dataset: List of tensors with shape (n, 6) where n can vary
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Pre-compute statistics
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute dataset statistics."""
        self.length_counts = defaultdict(int)
        for data in self.dataset:
            n = data.shape[0]
            self.length_counts[n] += 1
        
        self.total_samples = len(self.dataset)
        self.num_batches = 0
        
        for n, count in self.length_counts.items():
            num_full = count // self.batch_size
            has_trailing = (count % self.batch_size) > 0
            self.num_batches += num_full + (1 if has_trailing else 0)
    
    def __len__(self):
        """Return the number of batches."""
        return self.num_batches
    
    def __iter__(self):
        """Return an iterator over batches."""
        return create_dataloader(self.dataset, self.batch_size, self.shuffle)
    
    def get_stats(self):
        """Return dataset statistics."""
        stats = {
            'total_samples': self.total_samples,
            'num_batches': self.num_batches,
            'batch_size': self.batch_size,
            'length_distribution': dict(self.length_counts)
        }
        return stats
    


###############################################################################
# Config loading
###############################################################################


def load_model_params(json_path: str) -> dict:
    with open(json_path, "r") as f:
        params = json.load(f)

    # Convert any stringified tuples to tuples (e.g., for pulse_space ranges)
    if "pulse_space" in params:
        for k, v in params["pulse_space"].items():
            params["pulse_space"][k] = tuple(v)

    print(f"Parameters: {list(params.keys())}")

    return params

    

###############################################################################
# Driver Code
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Train composite pulse model")
    parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="weights/single_qubit_control/weights", help="Path to save model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debugging mode with smaller dataset")
    args = parser.parse_args()


    DEBUGGING = args.debug


    # Load model parameters from external JSON
    current_directory = os.path.dirname(__file__)
    model_params = load_model_params(f"{current_directory}/model_config.json")  
    model = UnitaryControlTransformer(**model_params)


    trainer_params = {
        "model": model,
        "unitary_generator": batched_unitary_generator,
        "fidelity_fn": fidelity,
        "loss_fn": sharp_loss,
    }

    trainer = UniversalModelTrainer(**trainer_params)

    if not DEBUGGING:
        train_size = 100000
        eval_size = 20000
    else:
        train_size = 2000
        eval_size = 400

    max_N = 4
    max_delta = 1.5

    print(f"Training with dataset size {train_size}, eval size {eval_size}, max_N {max_N}, max_delta {max_delta}")

    train_dataset = build_SU2_dataset(dataset_size=train_size, max_N=max_N, max_delta=max_delta)
    eval_dataset = build_SU2_dataset(dataset_size=eval_size, max_N=max_N, max_delta=max_delta)

    batch_size = args.batch_size

    train_dataloader = SU2DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = SU2DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)


    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=args.num_epoch,
        save_path=args.save_path,
        plot=True
    )



if __name__ == "__main__":
    torch.manual_seed(0)
    main()