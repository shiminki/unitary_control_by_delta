import math
from typing import Callable, Dict, Sequence, Tuple, Optional, List, Union
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import sys

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


from model.unitary_control_transformer import UnitaryControlTransformer


class UniversalModelTrainer:
    """Trainer for UnitaryControlTransformer with delta-range aware training."""

    def __init__(
        self,
        model: UnitaryControlTransformer,
        unitary_generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        fidelity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        monte_carlo: int = 1000,
        epsilon_std: float = 0.05,
        device: str = None,
    ) -> None:
        """
        Args:
            model: UnitaryControlTransformer
            unitary_generator: Function (pulses, errors) -> U_out where
                - pulses: (B, max_pulses, param_dim)
                - errors: (B, 2) with (delta, epsilon)
                - U_out: (B, 2, 2)
            fidelity_fn: Function (U_out, U_target) -> fidelity scores
            loss_fn: Optional loss function, defaults to 1 - mean(fidelity)
            optimizer: Optional optimizer, defaults to Adam(lr=3e-5)
            monte_carlo: Number of Monte Carlo samples per batch
            epsilon_std: Standard deviation for Gaussian noise epsilon
            device: Training device
        """
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
        self.model = model.to(device)
        self.unitary_generator = unitary_generator
        self.fidelity_fn = fidelity_fn
        self.loss_fn = loss_fn or (lambda U_out, U_target: 1.0 - self.fidelity_fn(U_out, U_target).mean())
        self.monte_carlo = monte_carlo
        self.epsilon_std = epsilon_std
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample Monte Carlo errors and determine corresponding target unitaries.
        
        Args:
            data_batch: (batch_size, n, 6) where each row is 
                       (delta_start, delta_end, n_x, n_y, n_z, theta)
        
        Returns:
            errors: (batch_size * monte_carlo, 2) with (delta, epsilon)
            targets: (batch_size * monte_carlo, 2, 2) target unitaries
            data_batch_repeated: (batch_size * monte_carlo, n, 6) for model input
        """
        batch_size, n, _ = data_batch.shape
        total_samples = batch_size * self.monte_carlo
        
        # Repeat data_batch for monte carlo samples
        # Shape: (batch_size, monte_carlo, n, 6)
        data_expanded = data_batch.unsqueeze(1).repeat(1, self.monte_carlo, 1, 1)
        # Reshape to (batch_size * monte_carlo, n, 6)
        data_repeated = data_expanded.reshape(total_samples, n, 6)
        
        # Extract delta ranges: (batch_size, monte_carlo, n, 2)
        delta_ranges = data_expanded[..., :2]  # (batch_size, monte_carlo, n, 2)
        
        # Sample deltas uniformly from the intervals
        # For each sample, pick one of the n intervals and sample uniformly
        errors = torch.zeros(total_samples, 2, device=self.device)
        targets = torch.zeros(total_samples, 2, 2, dtype=torch.complex128, device=self.device)
        
        for b in range(batch_size):
            for m in range(self.monte_carlo):
                idx = b * self.monte_carlo + m
                
                # Randomly select one interval from n intervals
                j = torch.randint(0, n, (1,)).item()
                
                # Sample delta uniformly from [delta_j_start, delta_j_end]
                delta_start = delta_ranges[b, m, j, 0].item()
                delta_end = delta_ranges[b, m, j, 1].item()
                delta = torch.rand(1).item() * (delta_end - delta_start) + delta_start
                
                # Sample epsilon from Gaussian
                epsilon = torch.randn(1).item() * self.epsilon_std
                
                errors[idx, 0] = delta
                errors[idx, 1] = epsilon
                
                # Get target unitary from j-th rotation vector
                n_x = data_batch[b, j, 2].item()
                n_y = data_batch[b, j, 3].item()
                n_z = data_batch[b, j, 4].item()
                theta = data_batch[b, j, 5].item()
                
                # Construct target unitary: exp(-i * theta/2 * (n_x*X + n_y*Y + n_z*Z))
                targets[idx] = self._rotation_unitary(n_x, n_y, n_z, theta)
        
        return errors, targets, data_repeated
    
    @staticmethod
    def _rotation_unitary(n_x: float, n_y: float, n_z: float, theta: float) -> torch.Tensor:
        """
        Construct rotation unitary exp(-i * theta/2 * n·σ).
        
        Returns:
            (2, 2) complex unitary matrix
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
        
        data_batch = data_batch.to(self.device)
        batch_size = data_batch.shape[0]
        
        # Get pulses from model (single forward pass)
        # pulses: (batch_size, max_pulses, param_dim)
        pulses = self.model(data_batch)
        
        # Sample Monte Carlo errors and targets
        # errors: (batch_size * monte_carlo, 2)
        # targets: (batch_size * monte_carlo, 2, 2)
        # data_repeated: (batch_size * monte_carlo, n, 6)
        errors, targets, data_repeated = self.sample_monte_carlo_batch(data_batch)
        
        # Repeat pulses for Monte Carlo samples
        # pulses_mc: (batch_size * monte_carlo, max_pulses, param_dim)
        pulses_mc = pulses.repeat_interleave(self.monte_carlo, dim=0)
        
        # Generate output unitaries
        # U_out: (batch_size * monte_carlo, 2, 2)
        U_out = self.unitary_generator(pulses_mc, errors)
        
        # Compute loss
        loss = self.loss_fn(U_out, targets)
        
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
        errors, targets, data_repeated = self.sample_monte_carlo_batch(data_batch)
        
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
    ) -> None:
        """
        Train the model using the provided dataloaders.
        
        Args:
            train_dataloader: Dataloader yielding (batch_size, n, 6) batches
            eval_dataloader: Dataloader for evaluation
            epochs: Number of training epochs
            save_path: Path to save checkpoints
            plot: Whether to plot training curves
        """
        
        self.model.to(self.device)
        
        fidelity_history = []
        loss_history = []
        
        with tqdm(total=epochs, desc="Training", dynamic_ncols=True) as pbar:
            for epoch in range(1, epochs + 1):
                # Training phase
                train_losses = []
                train_fids = []
                
                # Use a tqdm progress bar for the training dataloader to show per-epoch ETA
                train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} training", leave=False)
                running_losses = []
                running_fids = []
                for batch in train_bar:
                    loss, fid = self.train_epoch(batch)
                    train_losses.append(loss)
                    train_fids.append(fid)
                    running_losses.append(loss)
                    running_fids.append(fid)

                    # update train_bar postfix with running averages
                    avg_loss = np.mean(running_losses)
                    avg_fid = np.mean(running_fids)
                    train_bar.set_postfix({"loss": f"{avg_loss:.4f}", "fid": f"{avg_fid:.4f}"})
                
                mean_train_loss = np.mean(train_losses)
                mean_train_fid = np.mean(train_fids)
                
                # Evaluation phase (show per-epoch ETA with tqdm)
                eval_fids = []
                eval_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch} eval", leave=False)
                eval_running = []
                for batch in eval_bar:
                    eval_fid = self.evaluate(batch)
                    eval_fids.append(eval_fid)
                    eval_running.append(eval_fid)
                    eval_bar.set_postfix({"eval_fid": f"{np.mean(eval_running):.4f}"})
                
                mean_eval_fid = np.mean(eval_fids)
                
                # Track best model
                if mean_eval_fid > self.best_fidelity:
                    self.best_fidelity = mean_eval_fid
                    self.best_state = {
                        k: v.detach().cpu().clone() 
                        for k, v in self.model.state_dict().items()
                    }
                
                # Update progress bar
                pbar.set_postfix({
                    "epoch": epoch,
                    "train_loss": f"{mean_train_loss:.4f}",
                    "train_fid": f"{mean_train_fid:.4f}",
                    "eval_fid": f"{mean_eval_fid:.4f}",
                    "best": f"{self.best_fidelity:.4f}",
                })
                pbar.update(1)
                
                fidelity_history.append(mean_eval_fid)
                loss_history.append(mean_train_loss)


                if save_epoch is not None and epoch % save_epoch == 0 and save_path is not None:
                    epoch_save_path = Path(save_path).parent / f"{Path(save_path).stem}_epoch{epoch}.pt"
                    self._save_weight(epoch_save_path)
        
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

