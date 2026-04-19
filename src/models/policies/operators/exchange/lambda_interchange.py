"""
Lambda-interchange operator (vectorized).

Lambda-interchange is a systematic exploration of the cross-exchange neighborhood
with segment lengths ranging from 0 to λ. This is a powerful operator that
generalizes many simpler moves.
"""

from typing import Optional

import torch

from .cross_exchange import vectorized_cross_exchange


def vectorized_lambda_interchange(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    lambda_max: int = 2,
    max_iterations: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Vectorized λ-interchange local search across a batch of tours using PyTorch.

    λ-interchange systematically explores the cross-exchange neighborhood by trying
    all combinations of segment lengths from 0 to λ_max. This creates a rich
    neighborhood that includes:
    - λ=0: One-sided relocations (equivalent to relocate operator)
    - λ=1: Single node swaps and relocations
    - λ=2: Two-node segment exchanges
    - λ=k: k-node segment exchanges

    The operator iteratively applies cross-exchange with all segment length combinations
    until no improvement is found. This is more thorough than a single cross-exchange
    call but also more computationally expensive.

    Algorithm:
    1. For λ_a in [0, λ_max]:
        For λ_b in [0, λ_max]:
            Skip if both are 0 (no-op)
            Apply cross-exchange with segment lengths (λ_a, λ_b)
            If improvement found, restart from λ=0
    2. Repeat until no improvement found in full sweep

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        capacities: Vehicle capacities [B] or scalar (optional, for capacity checks)
        wastes: Node wastes [B, N+1] or [N+1] (optional, for capacity checks)
        lambda_max: Maximum segment length to consider (default: 2)
            Higher values = larger neighborhood but slower
        max_iterations: Maximum number of full neighborhood sweeps (default: 50)
        generator (Optional[torch.Generator]): PyTorch generator for random number generation.

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours should include depot as node 0
        - For single-route problems, this operator has no effect
        - This is a wrapper around vectorized_cross_exchange
        - Complexity: O(λ_max² × N⁴) per sweep
        - More expensive than single cross-exchange but finds better solutions
        - Recommended λ_max values: 1-3 (2 is good balance)

    Example:
        >>> tours = torch.tensor([[0, 3, 1, 0, 2, 4, 0]])  # Two routes
        >>> dist = torch.rand(5, 5)  # Distance matrix
        >>> improved = vectorized_lambda_interchange(tours, dist, lambda_max=2)
    """
    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

    # Handle distance_matrix expansion
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0)

    B, N = tours.shape
    if N < 4:  # Too small for lambda-interchange
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Main loop: try all segment length combinations
    for _iteration in range(max_iterations):
        improved_this_sweep = False

        # Systematically explore all (λ_a, λ_b) combinations
        for seg_a_len in range(lambda_max + 1):
            for seg_b_len in range(lambda_max + 1):
                # Skip no-op case
                if seg_a_len == 0 and seg_b_len == 0:
                    continue

                # Store initial tours to detect improvement
                tours_before = tours.clone()

                # Apply cross-exchange with this segment length combination
                tours = vectorized_cross_exchange(
                    tours=tours,
                    distance_matrix=distance_matrix,
                    capacities=capacities,
                    wastes=wastes,
                    max_segment_len=max(seg_a_len, seg_b_len),
                    max_iterations=1,  # Single iteration per combination
                )

                # Check if any improvement occurred
                if not torch.equal(tours, tours_before):
                    improved_this_sweep = True
                    # First-improvement strategy: restart search after improvement
                    break

            if improved_this_sweep:
                break  # Restart from λ=0

        # If no improvement in full sweep, we're done
        if not improved_this_sweep:
            break

    return tours if is_batch else tours.squeeze(0)
