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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from model.UnitaryControlTransformer import UnitaryControlTransformer
from model.UniversalModelTrainer import UniversalModelTrainer

from util.dataset.delta_control_SU2_dataset import build_SU2_dataset
from util.dataset.SU2DataLoader import SU2DataLoader
from util.fidelity.fidelity import fidelity_single_qubit_control
from util.loss_function.loss_function import sharp_loss
from util.unitary_generator.unitary_generators import batched_unitary_generator_single_qubit
from util.error_sampler.error_sampler import single_qubit_error_sampler


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


"""

Debug driver code
python train/train_single_qubit_control/train.py --save_path "test_weight/" --debug true --batch_size 4 --save_epoch 2 --num_epoch 10 --checkpoint_path "train/train_single_qubit_control/checkpoint/vanila_pretrain_iter_1_epoch18.pt"

"""


def main():
    parser = argparse.ArgumentParser(description="Train composite pulse model")
    parser.add_argument("--num_epoch", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--save_path", type=str, default="weights/single_qubit_control/weights", help="Path to save model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--debug", type=bool, default=False, help="Enable debugging mode with smaller dataset")
    parser.add_argument("--save_epoch", type=int, default=None, help="Save model every N epochs (default: None)")
    parser.add_argument("--monte_carlo", type=int, default=1000, help="Number of Monte Carlo samples")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint to load")
    args = parser.parse_args()

    DEBUGGING = args.debug

    # Load model parameters from external JSON
    current_directory = os.path.dirname(__file__)
    model_params = load_model_params(f"{current_directory}/model_config.json")  
    model = UnitaryControlTransformer(**model_params)

    if args.checkpoint_path is not None:
        print(f"Loading model weights from {args.checkpoint_path}")
        model.load_state_dict(torch.load(args.checkpoint_path))

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if not DEBUGGING:
        train_size = 100000
        eval_size = 20000
        monte_carlo = args.monte_carlo
    else:
        train_size = 200
        eval_size = 40
        monte_carlo = 100  # Smaller for debugging
        device = "cpu"  # Force CPU in debug mode

    # Fixed: use string "device" as key
    trainer_params = {
        "model": model,
        "unitary_generator": batched_unitary_generator_single_qubit,
        "error_sampler": single_qubit_error_sampler,
        "fidelity_fn": fidelity_single_qubit_control,
        "loss_fn": sharp_loss,
        "device": device, 
        "monte_carlo": monte_carlo
    }

    trainer = UniversalModelTrainer(**trainer_params)

    max_N = 4
    max_delta = 1.5

    print(f"Training with dataset size {train_size}, eval size {eval_size}, max_N {max_N}, max_delta {max_delta}")
    print(f"Monte Carlo samples: {monte_carlo}")

    train_dataset = build_SU2_dataset(dataset_size=train_size, max_N=max_N, max_delta=max_delta)
    eval_dataset = build_SU2_dataset(dataset_size=eval_size, max_N=max_N, max_delta=max_delta)

    batch_size = args.batch_size

    train_dataloader = SU2DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = SU2DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=args.num_epoch,
        save_path=args.save_path,
        plot=True,
        save_epoch=args.save_epoch,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    main()


