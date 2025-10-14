from typing import List
import torch
from collections import defaultdict
from random import shuffle
from util.dataset.delta_control_SU2_dataset import create_dataloader


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
    
