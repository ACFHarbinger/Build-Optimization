"""cluster_removal.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import cluster_removal
"""

from typing import Optional

import torch


def vectorized_cluster_removal(tours, dist_matrix, n_remove, generator: Optional[torch.Generator] = None):
    """
    Removes a cluster of spatially related nodes.
    """
    B, N = tours.size()
    device = tours.device

    # 1. Pick a random seed node for each batch
    seed_idx = torch.randint(0, N, (B,), device=device, generator=generator)
    seed_nodes = torch.gather(tours, 1, seed_idx.unsqueeze(1)).squeeze(1)

    # 2. Find distances from seed nodes to all other nodes
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    tour_distances = dist_matrix[batch_indices, seed_nodes.unsqueeze(1), tours]

    # 3. Take n_remove nearest nodes
    _, remove_idx = torch.topk(tour_distances, n_remove, dim=1, largest=False)

    # 4. Create mask and collapse
    mask = torch.ones_like(tours, dtype=torch.bool)
    mask[batch_indices[:, :n_remove], remove_idx] = False

    removed_nodes = torch.gather(tours, 1, remove_idx)

    # Compress tours (remove marked nodes)
    # Since we remove exactly n_remove nodes from each row, we can reshape safely
    new_tours = tours[mask].view(B, N - n_remove)

    return new_tours, removed_nodes
