import torch
import math
from typing import List, Tuple, Iterator
from collections import defaultdict



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









