import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List, Union
from pathlib import Path
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from model.UnitaryControlTransformer import UnitaryControlTransformer


class UniversalModelTrainer:
    """
    Universal trainer for UnitaryControlTransformer supporting multiple qubit systems.
    
    Supports:
    - Single qubit control with error shape (N, 2): [delta, epsilon]
    - Two-qubit entangled control with error shape (N, 4): [delta_sys, epsilon, delta_anc, coupling_error]
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
                - For two qubits: pulses (B, L, 5), errors (B, 4)
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
        
        # Timing statistics
        self.batch_times = []
        self.avg_batch_time = None

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
            targets: (batch_size * monte_carlo, d, d) target unitaries
        """
        batch_size, n, _ = data_batch.shape
        total_samples = batch_size * self.monte_carlo
        
        # Move data_batch to device if not already, for rotation vec access
        data_batch = data_batch.to(self.device)
        
        # Sample interval indices
        # interval_indices: (batch_size, monte_carlo)
        interval_indices = torch.randint(0, n, size=(batch_size, self.monte_carlo), device=self.device)
        
        # Reshape interval_indices to (total_samples,) and create batch indices
        # linear_indices: (total_samples,) - the index j in (b, j, 6)
        linear_indices = interval_indices.flatten() 
        # batch_indices: (total_samples,) - the index b in (b, j, 6)
        batch_indices = torch.arange(batch_size, device=self.device).repeat_interleave(self.monte_carlo)

        # Get the (n_x, n_y, n_z, theta) for all total_samples at once
        # rotation_vecs_mc: (total_samples, 4)
        # Note: data_batch[batch_indices, linear_indices, 2:] is the most concise way 
        # to implement this 'fancy indexing' in PyTorch for this specific case.
        rotation_vecs_mc = data_batch[batch_indices, linear_indices, 2:] 
        
        # Generate target unitaries (Vectorized)
        targets = self._rotation_unitary(
            rotation_vecs_mc[..., 0], # n_x
            rotation_vecs_mc[..., 1], # n_y
            rotation_vecs_mc[..., 2], # n_z
            rotation_vecs_mc[..., 3]  # theta
        )
        # targets is now (total_samples, 2, 2) on the device
        
        # Sample errors using the provided error_sampler
        # errors: appropriate shape for the system, on device
        errors = self.error_sampler(total_samples, self.device)
        
        return errors, targets
    
    def _rotation_unitary(
        self, 
        n_x: torch.Tensor, 
        n_y: torch.Tensor, 
        n_z: torch.Tensor, 
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized construction of rotation unitary exp(-i * theta/2 * n·σ).
        
        Args:
            n_x, n_y, n_z, theta: Tensors of shape (total_samples,) on device.
        
        Returns:
            (total_samples, 2, 2) complex unitary matrix on device.
        """
        # Ensure all inputs are complex for stability
        theta_half = theta / 2.0
        
        # Cosine and Sine of rotation angle
        c = torch.cos(theta_half)
        s = torch.sin(theta_half)
        
        # Components of -i * sin(theta/2) * (n_x*X + n_y*Y + n_z*Z)
        i_s = -1j * s.to(torch.complex128) # Complex scalar: (-i * sin(theta/2))
        
        # The complex components needed for off-diagonal
        i_s_nx = i_s * n_x.to(torch.complex128)
        i_s_ny = i_s * n_y.to(torch.complex128)
        i_s_nz = i_s * n_z.to(torch.complex128)

        # Build the 2x2 matrix using torch.stack and torch.cat
        # U = cos(a)I - i*sin(a)*(n_x*X + n_y*Y + n_z*Z)
        
        # Diagonal elements: c +/- i_s_nz
        u00 = c.to(torch.complex128) + i_s_nz
        u11 = c.to(torch.complex128) - i_s_nz
        
        # Off-diagonal elements: -i*s*(n_x - i*n_y) and -i*s*(n_x + i*n_y)
        # U01: -i*s*n_x + (-i*s)*(-i*n_y) = -i*s*n_x - s*n_y = -s(i*n_x + n_y)
        u01 = i_s_nx + i_s_ny * 1j # -i*s*n_x + s*n_y
        
        # U10: -i*s*n_x - (-i*s)*(-i*n_y) = -i*s*n_x + s*n_y
        u10 = i_s_nx - i_s_ny * 1j # -i*s*n_x - s*n_y
        
        # Stack the components: (total_samples, 2, 2)
        U = torch.stack([
            torch.stack([u00, u01], dim=-1),
            torch.stack([u10, u11], dim=-1)
        ], dim=-2)
        
        return U.contiguous() # Ensure memory is contiguous

    # ------------------------------------------------------------------
    # Training loop with timing
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
        
        # Compute loss - Handle different loss function signatures
        try:
            # Try calling with fidelity_fn as keyword argument (for sharp_loss, etc.)
            loss = self.loss_fn(U_out, targets, fidelity_fn=self.fidelity_fn)
        except TypeError:
            # If that fails, try without fidelity_fn (for lambda functions)
            try:
                loss = self.loss_fn(U_out, targets)
            except Exception as e:
                print(f"Error in loss function: {e}")
                print(f"U_out shape: {U_out.shape}, targets shape: {targets.shape}")
                raise
        
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
    # Top-level training orchestrator with enhanced progress tracking
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
        verbose_batch: bool = True,  # New parameter for batch-level logging
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
            verbose_batch: If True, show detailed batch progress
        """
        self.model.to(self.device)
        
        fidelity_history = []
        loss_history = []
        
        # Print header
        print("\n" + "=" * 90)
        print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Train Fid':>10} | {'Eval Fid':>10} | {'Best Fid':>10} | {'Time':>8}")
        print("=" * 90)
        
        total_train_batches = len(list(train_dataloader))
        total_eval_batches = len(list(eval_dataloader))
        
        print(f"\nTraining Details:")
        print(f"  - Training batches per epoch: {total_train_batches}")
        print(f"  - Evaluation batches per epoch: {total_eval_batches}")
        print(f"  - Monte Carlo samples per batch: {self.monte_carlo}")
        print(f"  - Verbose batch logging: {'Enabled' if verbose_batch else 'Disabled'}")
        print()
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_losses = []
            train_fids = []
            
            # Create progress bar with enhanced information
            train_pbar = tqdm(
                train_dataloader, 
                desc=f"Epoch {epoch}/{epochs} [Train]",
                total=total_train_batches,
                ncols=120,
                leave=verbose_batch
            )
            
            try:
                for batch_idx, batch in enumerate(train_pbar):
                    batch_start_time = time.time()
                    
                    # Perform training step
                    loss, fid = self.train_epoch(batch)
                    
                    batch_time = time.time() - batch_start_time
                    self.batch_times.append(batch_time)
                    
                    # Update running average (use last 20 batches)
                    recent_times = self.batch_times[-20:] if len(self.batch_times) > 20 else self.batch_times
                    self.avg_batch_time = np.mean(recent_times)
                    
                    train_losses.append(loss)
                    train_fids.append(fid)
                    
                    # Calculate ETA
                    remaining_batches = total_train_batches - (batch_idx + 1)
                    eta_seconds = remaining_batches * self.avg_batch_time if self.avg_batch_time else 0
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    
                    # Update progress bar with detailed information
                    train_pbar.set_postfix({
                        'Loss': f'{loss:.4f}',
                        'Fid': f'{fid:.4f}',
                        'Batch_Time': f'{batch_time:.1f}s',
                        'Avg_Time': f'{self.avg_batch_time:.1f}s',
                        'ETA': eta_str
                    })
                    
                    # Verbose batch logging
                    if verbose_batch and (batch_idx % 10 == 0 or batch_idx == 0):
                        print(f"\n  Batch {batch_idx+1}/{total_train_batches}: "
                              f"Loss={loss:.6f}, Fid={fid:.6f}, "
                              f"Time={batch_time:.2f}s, "
                              f"GPU_Mem={torch.cuda.memory_allocated()/1e9:.2f}GB")
                        
            except Exception as e:
                print(f"\nError during training at epoch {epoch}, batch {batch_idx+1}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                
                # Try to save current state before exiting
                if save_path is not None:
                    emergency_save = Path(save_path).parent / f"emergency_epoch{epoch}_batch{batch_idx}.pt"
                    try:
                        self._save_weight_checkpoint(emergency_save)
                        print(f"Emergency checkpoint saved to {emergency_save}")
                    except:
                        print("Failed to save emergency checkpoint")
                raise
            
            mean_train_loss = np.mean(train_losses)
            mean_train_fid = np.mean(train_fids)
            
            # Print training phase summary
            print(f"\n  Training phase completed: Avg Loss={mean_train_loss:.6f}, Avg Fid={mean_train_fid:.6f}")
            
            # Evaluation phase with progress tracking
            eval_fids = []
            eval_pbar = tqdm(
                eval_dataloader, 
                desc=f"Epoch {epoch}/{epochs} [Eval]",
                total=total_eval_batches,
                ncols=120,
                leave=False
            )
            
            for batch_idx, batch in enumerate(eval_pbar):
                batch_start_time = time.time()
                
                eval_fid = self.evaluate(batch)
                eval_fids.append(eval_fid)
                
                batch_time = time.time() - batch_start_time
                
                # Update progress bar
                eval_pbar.set_postfix({
                    'Fid': f'{eval_fid:.4f}',
                    'Batch_Time': f'{batch_time:.1f}s',
                    'Progress': f'{batch_idx+1}/{total_eval_batches}'
                })
            
            mean_eval_fid = np.mean(eval_fids)
            
            # Track best model
            if mean_eval_fid > self.best_fidelity:
                self.best_fidelity = mean_eval_fid
                self.best_state = {
                    k: v.detach().cpu().clone() 
                    for k, v in self.model.state_dict().items()
                }
                print(f"  ★ New best model! Fidelity: {self.best_fidelity:.6f}")
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch metrics
            print(f"{epoch:6d} | {mean_train_loss:12.6f} | {mean_train_fid:10.6f} | "
                  f"{mean_eval_fid:10.6f} | {self.best_fidelity:10.6f} | {epoch_time:7.1f}s")
            
            # Estimate total remaining time
            avg_epoch_time = epoch_time
            remaining_epochs = epochs - epoch
            total_eta = remaining_epochs * avg_epoch_time
            print(f"  Estimated time remaining: {str(timedelta(seconds=int(total_eta)))}")
            
            fidelity_history.append(mean_eval_fid)
            loss_history.append(mean_train_loss)
            
            # Save checkpoint at specified interval
            if save_epoch is not None and epoch % save_epoch == 0 and save_path is not None:
                epoch_save_path = Path(save_path).parent / f"{Path(save_path).stem}_epoch{epoch}.pt"
                self._save_weight_checkpoint(epoch_save_path)
            
            # Clear some GPU cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("=" * 90)
        print(f"Training completed! Best fidelity: {self.best_fidelity:.6f}")
        print(f"Average batch processing time: {self.avg_batch_time:.2f}s")
        print()
        
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