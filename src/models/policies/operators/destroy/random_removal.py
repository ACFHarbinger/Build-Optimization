"""random_removal.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import random_removal
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


def vectorized_random_removal(
    tours: Tensor, n_remove: int, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """
    Vectorized random removal of nodes from tours.

    Args:
        tours (Tensor): Batch of tours (B, N).
        n_remove (int): Number of nodes to remove.
        generator (Optional[torch.Generator]): PyTorch generator for random number generation.

    Returns:
        Tuple[Tensor, Tensor]:
            - Modified tours (B, N-n_remove) (or padded with clean 0s if needed)
            - Removed nodes (B, n_remove)
    """
    B, N = tours.shape
    device = tours.device

    # Identify valid customers (non-zero)
    customers_mask = tours > 0

    # Create random scores for sorting
    scores = torch.rand((B, N), device=device, generator=generator)
    scores[~customers_mask] = -1.0  # Depots/Padding have low score

    # Get indices of top k scores
    _, remove_indices = torch.topk(scores, k=n_remove, dim=1)  # (B, n_remove)

    # Gather removed nodes
    removed_nodes = torch.gather(tours, 1, remove_indices)

    # Create mask and collapse
    remove_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    remove_mask.scatter_(1, remove_indices, True)

    keep_mask = ~remove_mask

    # We need to collapse the kept nodes.
    # Since we remove exactly n_remove nodes from each row,
    # the number of kept nodes is N - n_remove.
    new_tours = tours[keep_mask].view(B, N - n_remove)

    return new_tours, removed_nodes
