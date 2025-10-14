import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List, Union
from pathlib import Path
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from model.UnitaryControlTransformer import UnitaryControlTransformer


class UniversalModelTrainer:
    """
    Universal trainer for UnitaryControlTransformer supporting multiple qubit systems.
    
    Supports:
    - Single qubit control with error shape (N, 2): [delta, epsilon]
    - Two-qubit entangled control with error shape (4, N): [delta_sys, epsilon, delta_anc, coupling_error]
    """

    def __init__(
        self,
        model: UnitaryControlTransformer,
        unitary_generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        error_sampler: Callable[[int, torch.device], torch.Tensor],
        *,
        fidelity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        monte_carlo: int = 10000,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model: UnitaryControlTransformer
            unitary_generator: Function (pulses, errors) -> U_out where
                - For single qubit: pulses (B, L, 3), errors (B, 2)
                - For two qubits: pulses (B, L, 5), errors (4, B)
            error_sampler: Function (batch_size, device) -> errors
                - Should return appropriate shape for the system
            fidelity_fn: Function (U_out, U_target) -> fidelity scores
            loss_fn: Optional loss function, defaults to 1 - mean(fidelity)
            optimizer: Optional optimizer, defaults to Adam(lr=3e-5)
            monte_carlo: Number of Monte Carlo samples per batch
            device: Training device
        """
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        self.model = model.to(device)
        self.unitary_generator = unitary_generator
        self.error_sampler = error_sampler
        self.fidelity_fn = fidelity_fn
        self.loss_fn = loss_fn or (lambda U_out, U_target: 1.0 - self.fidelity_fn(U_out, U_target).mean())
        self.monte_carlo = monte_carlo
        self.device = device

        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters(), lr=3e-5)

        # State tracking
        self.best_state: dict[str, torch.Tensor] | None = None
        self.best_fidelity: float = 0.0

    # ------------------------------------------------------------------
    # Helper: Sample delta and determine target unitary
    # ------------------------------------------------------------------
    
    def sample_monte_carlo_batch(
        self, 
        data_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample Monte Carlo errors and determine corresponding target unitaries.
        
        Args:
            data_batch: (batch_size, n, 6) where each row is 
                       (delta_start, delta_end, n_x, n_y, n_z, theta)
        
        Returns:
            errors: Appropriate shape for the system
                - Single qubit: (batch_size * monte_carlo, 2)
                - Two qubits: (4, batch_size * monte_carlo)
            targets: (batch_size * monte_carlo, d, d) target unitaries
                - d=2 for single qubit, d=4 for two qubits
        """
        batch_size, n, _ = data_batch.shape
        total_samples = batch_size * self.monte_carlo
        
        # Move data_batch to device if not already
        data_batch = data_batch.to(self.device)
        
        # Extract delta ranges and rotation vectors for sampling
        delta_ranges_cpu = data_batch[..., :2].cpu().numpy()  # (batch_size, n, 2)
        rotation_vecs = data_batch[..., 2:].cpu().numpy()  # (batch_size, n, 4)
        
        # Sample interval indices and delta values
        interval_indices = np.random.randint(0, n, size=(batch_size, self.monte_carlo))
        uniform_samples = np.random.rand(batch_size, self.monte_carlo)
        
        # Build target unitaries
        targets_list = []
        sampled_deltas = np.zeros((batch_size, self.monte_carlo))
        
        for b in range(batch_size):
            for m in range(self.monte_carlo):
                j = interval_indices[b, m]
                
                # Sample delta uniformly from [delta_j_start, delta_j_end]
                delta_start = delta_ranges_cpu[b, j, 0]
                delta_end = delta_ranges_cpu[b, j, 1]
                delta = uniform_samples[b, m] * (delta_end - delta_start) + delta_start
                sampled_deltas[b, m] = delta
                
                # Get target unitary from j-th rotation vector
                n_x, n_y, n_z, theta = rotation_vecs[b, j]
                U = self._rotation_unitary(n_x, n_y, n_z, theta)
                targets_list.append(U)
        
        # Stack targets and move to GPU
        targets = torch.stack(targets_list).to(self.device)
        
        # Sample errors using the provided error_sampler
        # The error_sampler should handle the specific error model
        errors = self.error_sampler(total_samples, self.device)
        
        
        return errors, targets
    
    @staticmethod
    def _rotation_unitary(n_x: float, n_y: float, n_z: float, theta: float) -> torch.Tensor:
        """
        Construct rotation unitary exp(-i * theta/2 * n·σ).
        
        Returns:
            (2, 2) complex unitary matrix on CPU (will be moved to GPU in batch)
        """
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        
        # exp(-i * theta/2 * n·σ) = cos(theta/2)*I - i*sin(theta/2)*(n_x*X + n_y*Y + n_z*Z)
        U = torch.tensor([
            [c - 1j * s * n_z, -1j * s * (n_x - 1j * n_y)],
            [-1j * s * (n_x + 1j * n_y), c + 1j * s * n_z]
        ], dtype=torch.cdouble)
        
        return U

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, data_batch: torch.Tensor) -> Tuple[float, float]:
        """
        One training step on a batch.
        
        Args:
            data_batch: (batch_size, n, 6) tensor
        
        Returns:
            loss: Scalar loss value
            fidelity: Mean fidelity across monte carlo samples
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Ensure data is on correct device
        data_batch = data_batch.to(self.device)
        batch_size = data_batch.shape[0]
        
        # Get pulses from model (single forward pass)
        # pulses: (batch_size, max_pulses, param_dim)
        pulses = self.model(data_batch)
        
        # Sample Monte Carlo errors and targets
        # errors: appropriate shape for system
        # targets: (batch_size * monte_carlo, d, d)
        errors, targets = self.sample_monte_carlo_batch(data_batch)
        
        # Repeat pulses for Monte Carlo samples
        # pulses_mc: (batch_size * monte_carlo, max_pulses, param_dim)
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)
        
        # Generate output unitaries
        # U_out: (batch_size * monte_carlo, d, d)
        U_out = self.unitary_generator(pulses_mc, errors)
        
        # Compute loss - FIXED: Pass fidelity_fn as kwarg
        if hasattr(self.loss_fn, '__self__'):
            # If loss_fn is a bound method, call it directly
            loss = self.loss_fn(U_out, targets)
        else:
            # If loss_fn is a function that needs fidelity_fn
            loss = self.loss_fn(U_out, targets, fidelity_fn=self.fidelity_fn)
        
        # Compute mean fidelity for monitoring
        with torch.no_grad():
            mean_fid = self.fidelity_fn(U_out, targets).mean().item()
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return float(loss.item()), mean_fid



    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, data_batch: torch.Tensor) -> float:
        """
        Evaluate mean fidelity on a batch (no gradients).
        
        Args:
            data_batch: (batch_size, n, 6) tensor
        
        Returns:
            mean_fidelity: Mean fidelity across monte carlo samples
        """
        self.model.eval()
        
        data_batch = data_batch.to(self.device)
        
        # Get pulses from model
        pulses = self.model(data_batch)
        
        # Sample Monte Carlo errors and targets
        errors, targets = self.sample_monte_carlo_batch(data_batch)
        
        # Repeat pulses for Monte Carlo samples
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)
        
        # Generate output unitaries
        U_out = self.unitary_generator(pulses_mc, errors)
        
        # Compute mean fidelity
        mean_fid = self.fidelity_fn(U_out, targets).mean().item()
        
        return mean_fid

    # ------------------------------------------------------------------
    # Top-level training orchestrator
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataloader,
        eval_dataloader,
        epochs: int = 100,
        save_path: str | Path | None = None,
        plot: bool = False,
        save_epoch: int = None,
        log_interval: int = 1,
    ) -> None:
        """
        Train the model using the provided dataloaders.
        
        Args:
            train_dataloader: Dataloader yielding (batch_size, n, 6) batches
            eval_dataloader: Dataloader for evaluation
            epochs: Number of training epochs
            save_path: Path to save checkpoints
            plot: Whether to plot training curves
            save_epoch: Save checkpoint every N epochs
            log_interval: Log metrics every N epochs
        """
        self.model.to(self.device)
        
        fidelity_history = []
        loss_history = []
        
        # Print header
        print("\n" + "=" * 90)
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Train Fid':>10} | {'Eval Fid':>10} | {'Best Fid':>10} | {'Time':>8}")
        print("=" * 90)
        
        import time
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_losses = []
            train_fids = []
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} [Train]", 
                            leave=False, ncols=80, disable=False):
                loss, fid = self.train_epoch(batch)
                train_losses.append(loss)
                train_fids.append(fid)
            
            mean_train_loss = np.mean(train_losses)
            mean_train_fid = np.mean(train_fids)
            
            # Evaluation phase
            eval_fids = []
            for batch in tqdm(eval_dataloader, desc=f"Epoch {epoch}/{epochs} [Eval]", 
                            leave=False, ncols=80, disable=False):
                eval_fid = self.evaluate(batch)
                eval_fids.append(eval_fid)
            
            mean_eval_fid = np.mean(eval_fids)
            
            # Track best model
            if mean_eval_fid > self.best_fidelity:
                self.best_fidelity = mean_eval_fid
                self.best_state = {
                    k: v.detach().cpu().clone() 
                    for k, v in self.model.state_dict().items()
                }
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics at specified interval
            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"{epoch:6d} | {mean_train_loss:12.6f} | {mean_train_fid:10.6f} | "
                      f"{mean_eval_fid:10.6f} | {self.best_fidelity:10.6f} | {epoch_time:7.1f}s")
            
            fidelity_history.append(mean_eval_fid)
            loss_history.append(mean_train_loss)
            
            # Save checkpoint at specified interval
            if save_epoch is not None and epoch % save_epoch == 0 and save_path is not None:
                epoch_save_path = Path(save_path).parent / f"{Path(save_path).stem}_epoch{epoch}.pt"
                self._save_weight_checkpoint(epoch_save_path)
        
        print("=" * 90)
        print(f"Training completed! Best fidelity: {self.best_fidelity:.6f}\n")
        
        # Reload best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        # Save model
        if save_path is not None:
            self._save_weight(save_path)
        
        # Plot training curves
        if plot and save_path is not None:
            self._plot_training_curves(
                loss_history, 
                fidelity_history, 
                save_path
            )

    # ------------------------------------------------------------------
    # Persistence and plotting helpers
    # ------------------------------------------------------------------

    def _save_weight_checkpoint(self, path: str | Path) -> None:
        """Save a checkpoint (current state, not necessarily best)."""
        self.model.eval()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save current state
        current_state = {
            k: v.detach().cpu().clone() 
            for k, v in self.model.state_dict().items()
        }
        torch.save(current_state, str(path))
        print(f"    Checkpoint saved → {path}")

    def _save_weight(self, path: str | Path) -> None:
        """Save the best model weights."""
        self.model.eval()
        if self.best_state is None:
            raise RuntimeError("No trained weights recorded – call .train() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.best_state, str(path))
        print(f"Weights saved → {path}")

    def _plot_training_curves(
        self, 
        loss_history: List[float], 
        fidelity_history: List[float],
        save_path: str | Path
    ) -> None:
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs_range = range(1, len(loss_history) + 1)
        
        # Loss plot
        ax1.plot(epochs_range, loss_history, marker='o')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss vs Epoch")
        ax1.grid(True)
        
        # Fidelity plot
        ax2.plot(epochs_range, fidelity_history, marker='o', color='green')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Evaluation Fidelity")
        ax2.set_title("Evaluation Fidelity vs Epoch")
        ax2.grid(True)
        
        plt.tight_layout()
        
        fig_path = Path(save_path).parent / f"{Path(save_path).stem}_training_curves.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()
        print(f"Training curves saved → {fig_path}")

    @torch.no_grad()
    def get_average_fidelity(self, dataloader) -> float:
        """
        Compute average fidelity across all batches in dataloader.
        
        Args:
            dataloader: Dataloader yielding (batch_size, n, 6) batches
        
        Returns:
            mean_fidelity: Average fidelity
        """
        self.model.eval()
        self.model.to(self.device)
        
        fidelities = []
        for batch in dataloader:
            fid = self.evaluate(batch)
            fidelities.append(fid)
        
        return np.mean(fidelities)

