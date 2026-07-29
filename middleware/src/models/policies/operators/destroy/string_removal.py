"""
String removal operator (vectorized).

String removal (SISR - Slack Induction by String Removal) removes contiguous
sequences of nodes to create large spatial gaps that can be efficiently rearranged.
"""

from typing import Optional, Tuple

import torch


def vectorized_string_removal(
    tours: torch.Tensor,
    n_remove: int,
    max_string_len: int = 4,
    avg_string_len: float = 3.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized string removal across a batch of tours using PyTorch.

    String removal removes contiguous sequences (strings) of nodes from routes.
    The key insight is that removing adjacent customers creates a large contiguous
    "hole" in the route, providing more flexibility for reinsertion compared to
    random removal.

    Algorithm:
    1. While removed < n_remove:
        a. Select random seed node from tour
        b. Determine string length using geometric distribution
        c. Remove string starting at seed (up to max_string_len)
        d. Mark removed nodes
    2. Return tours with removed nodes marked as -1

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        n_remove: Number of nodes to remove from each tour
        max_string_len: Maximum length of a string to remove (default: 4)
        avg_string_len: Average string length for geometric distribution (default: 3.0)
        generator (Optional[torch.Generator]): PyTorch generator for random number generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Modified tours [B, N] with removed nodes replaced by padding (-1)
            - Removed nodes [B, n_remove] indices of removed nodes

    Note:
        - Tours should include depot as node 0 (depot not removed)
        - Removed nodes are marked as -1 in returned tours
        - String length follows geometric-like distribution: P(L=k) ~ (1-1/avg)^(k-1)
        - Stops early if not enough nodes available
    """
    device = tours.device

    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

    B, N = tours.shape

    # Initialize removed nodes tracking
    removed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    removed_list = torch.full((B, n_remove), -1, dtype=torch.long, device=device)
    removed_count = torch.zeros(B, dtype=torch.long, device=device)

    # Valid nodes (non-depot, non-padding)
    valid_mask = (tours > 0) & (tours < N)

    for b in range(B):
        remaining_to_remove = n_remove
        max_iter = n_remove * 3  # Prevent infinite loops
        iterations = 0

        while remaining_to_remove > 0 and iterations < max_iter:
            iterations += 1

            # Get available nodes (not yet removed, not depot)
            available_mask = valid_mask[b] & (~removed_mask[b])
            available_indices = torch.where(available_mask)[0]

            if len(available_indices) == 0:
                break

            # Select random seed
            seed_idx = available_indices[
                torch.randint(len(available_indices), (1,), device=device, generator=generator)
            ]

            # Determine string length using geometric distribution
            # L = 1 + geometric samples
            string_len = 1
            p_continue = 1.0 - 1.0 / avg_string_len

            while string_len < max_string_len and torch.rand(1, device=device, generator=generator).item() < p_continue:
                string_len += 1

            # Limit to remaining removal quota and tour length
            string_len = min(string_len, int(remaining_to_remove), int(N) - int(seed_idx.item()))

            # Remove string starting at seed
            for offset in range(string_len):
                node_idx = seed_idx + offset
                if node_idx < N and available_mask[node_idx]:
                    removed_mask[b, node_idx] = True
                    removed_list[b, removed_count[b]] = node_idx
                    removed_count[b] += 1
                    remaining_to_remove -= 1

                    if remaining_to_remove <= 0:
                        break

    # Create modified tours with removed nodes marked as -1
    modified_tours = tours.clone()
    for b in range(B):
        modified_tours[b, removed_mask[b]] = -1

    return (
        modified_tours if is_batch else modified_tours.squeeze(0),
        removed_list if is_batch else removed_list.squeeze(0),
    )
