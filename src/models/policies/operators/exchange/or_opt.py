"""
Or-opt local search operator (vectorized).

The Or-opt operator relocates chains of 1-3 consecutive nodes to better positions,
either within the same route or to different routes. This is particularly effective
for geographically clustered customers.
"""

from typing import Optional

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_or_opt(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    chain_lengths: tuple = (1, 2, 3),
    max_iterations: int = 100,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Vectorized Or-opt local search across a batch of tours using PyTorch.

    The Or-opt operator tries to relocate chains of consecutive nodes (typically
    1, 2, or 3 nodes) to better positions in the tour. This is a generalization
    of node relocation that considers longer sequences.

    For a chain of k nodes starting at position i:
    Original: ... -> a -> [c1 -> c2 -> ... -> ck] -> b -> ...
    Or-opt:   ... -> a -> b -> ... -> x -> [c1 -> c2 -> ... -> ck] -> y -> ...

    The algorithm evaluates:
    1. Removal gain = d(a,c1) + d(ck,b) - d(a,b)
    2. Insertion cost = d(x,c1) + d(ck,y) - d(x,y)
    3. Delta = insertion_cost - removal_gain

    This implementation processes batches in parallel for GPU acceleration.

    Algorithm:
    1. For each chain length k in [1, 2, 3]:
        a. For all valid chain positions in parallel:
            - Compute removal gain
            - For all insertion positions:
                - Compute insertion cost and delta
        b. Select best improvement for each tour
        c. Apply moves where delta < 0
    2. Repeat until no improvement or max_iterations

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        capacities: Vehicle capacities [B] or scalar (optional, for capacity checks)
        wastes: Node wastes [B, N+1] or [N+1] (optional, for capacity checks)
        chain_lengths: Tuple of chain lengths to try (default: (1, 2, 3))
        max_iterations: Maximum number of improvement iterations (default: 100)
        generator (Optional[torch.Generator]): PyTorch generator for random number generation.

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours should include depot as node 0
        - Capacity constraints are checked if capacities and wastes provided
        - Works with both batched and shared distance matrices
        - Stops early if no improvement found
        - Chain lengths > N/2 are automatically skipped
    """
    device = distance_matrix.device

    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

    # Handle distance_matrix expansion
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0)

    B, N = tours.shape
    if N < 4:  # Too small for or-opt
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Handle wastes if provided
    has_capacity = capacities is not None and wastes is not None
    if has_capacity:
        assert wastes is not None
        assert capacities is not None
        if wastes.dim() == 1:
            wastes = wastes.unsqueeze(0).expand(B, -1)
        if capacities.dim() == 0:
            capacities = capacities.unsqueeze(0).expand(B)

    batch_indices = torch.arange(B, device=device).view(B, 1)

    # Iterate through chain lengths
    for chain_len in chain_lengths:
        if chain_len >= N // 2:
            continue

        for _iteration in range(max_iterations):
            improved_any = False
            chain_starts = torch.arange(N - chain_len, device=device)
            n_chains = len(chain_starts)
            if n_chains == 0:
                break

            chain_starts_batch = chain_starts.unsqueeze(0).expand(B, -1)

            # Compute removal gain and chain wastes
            removal_gain, chain_wastes = _compute_removal_info(
                tours,
                chain_starts_batch,
                chain_len,
                distance_matrix,
                wastes,
                has_capacity,
                batch_indices,
                B,
                n_chains,
                N,
            )

            # Find best insertions
            best_delta, best_insert_pos = _find_best_insertions(
                B,
                n_chains,
                N,
                chain_len,
                chain_starts,
                tours,
                distance_matrix,
                removal_gain,
                chain_wastes,
                capacities,
                has_capacity,
                batch_indices,
                device,
            )

            # Find best chain to move for each batch instance
            best_chain_delta, best_chain_idx = torch.min(best_delta, dim=1)

            improved = best_chain_delta < -IMPROVEMENT_EPSILON
            if not improved.any():
                break

            improved_any = True
            tours = _apply_or_opt_moves(
                tours, improved, best_chain_idx, best_insert_pos, chain_starts, chain_len, B, N, device
            )

            if not improved_any:
                break

    return tours if is_batch else tours.squeeze(0)


def _compute_removal_info(
    tours, chain_starts, chain_len, dist_mat, wastes, has_capacity, batch_indices, B, n_chains, N
):
    """Computes removal gain and chain wastes."""
    prev_idx = (chain_starts - 1).clamp(min=0)
    next_idx = (chain_starts + chain_len).clamp(max=N - 1)

    prev_nodes = torch.gather(tours, 1, prev_idx)
    chain_first = torch.gather(tours, 1, chain_starts)
    chain_last = torch.gather(tours, 1, (chain_starts + chain_len - 1).clamp(max=N - 1))
    next_nodes = torch.gather(tours, 1, next_idx)

    b_exp = batch_indices.expand(B, n_chains)
    removal_gain = (
        dist_mat[b_exp, prev_nodes, chain_first]
        + dist_mat[b_exp, chain_last, next_nodes]
        - dist_mat[b_exp, prev_nodes, next_nodes]
    )

    chain_wastes = None
    if has_capacity:
        chain_wastes = torch.zeros(B, n_chains, device=tours.device)
        for k in range(chain_len):
            nodes = torch.gather(tours, 1, (chain_starts + k).clamp(max=N - 1))
            chain_wastes += torch.gather(wastes, 1, nodes)

    return removal_gain, chain_wastes


def _find_best_insertions(
    B, n_chains, N, chain_len, chain_starts, tours, dist_mat, rem_gain, chain_wastes, caps, has_cap, b_idx, device
):
    """Finds best insertion positions for all chains in batch."""
    best_delta = torch.full((B, n_chains), float("inf"), device=device)
    best_insert_pos = torch.zeros((B, n_chains), dtype=torch.long, device=device)
    b_exp = b_idx.expand(B, n_chains)

    # Get chain nodes for insertion cost calculation
    chain_first = torch.gather(tours, 1, chain_starts.unsqueeze(0).expand(B, -1))
    chain_last = torch.gather(tours, 1, (chain_starts + chain_len - 1).clamp(max=N - 1).unsqueeze(0).expand(B, -1))

    for insert_pos in range(N + 1):
        ins_prev = torch.full((B, n_chains), insert_pos - 1, device=device).clamp(min=0, max=N - 1)
        ins_next = torch.full((B, n_chains), insert_pos, device=device).clamp(min=0, max=N - 1)

        ins_prev_nodes = torch.gather(tours, 1, ins_prev)
        ins_next_nodes = torch.gather(tours, 1, ins_next)

        # Skip positions inside original chain
        skip_mask = (chain_starts.unsqueeze(0) <= insert_pos) & (insert_pos <= chain_starts.unsqueeze(0) + chain_len)

        ins_cost = (
            dist_mat[b_exp, ins_prev_nodes, chain_first]
            + dist_mat[b_exp, chain_last, ins_next_nodes]
            - dist_mat[b_exp, ins_prev_nodes, ins_next_nodes]
        )

        if has_cap:
            feasible = chain_wastes <= caps.unsqueeze(1)
            ins_cost = torch.where(feasible, ins_cost, torch.tensor(float("inf"), device=device))

        delta = torch.where(skip_mask, torch.tensor(float("inf"), device=device), ins_cost - rem_gain)

        better = delta < best_delta
        best_delta = torch.where(better, delta, best_delta)
        best_insert_pos = torch.where(better, ins_next, best_insert_pos)

    return best_delta, best_insert_pos


def _apply_or_opt_moves(tours, improved, best_chain_idx, best_insert_pos, chain_starts, chain_len, B, N, device):
    """Applies the best Or-opt moves found to the batch of tours."""
    for b in range(B):
        if improved[b]:
            idx = best_chain_idx[b].item()
            start = chain_starts[idx].item()
            pos = best_insert_pos[b, idx].item()

            chain = tours[b, start : start + chain_len].clone()
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[start : start + chain_len] = False
            remaining = tours[b][mask]

            adj_pos = pos - chain_len if pos > start + chain_len else pos
            new_tour = torch.cat([remaining[:adj_pos], chain, remaining[adj_pos:]])

            if new_tour.size(0) < N:
                new_tour = torch.cat([new_tour, torch.zeros(N - new_tour.size(0), dtype=tours.dtype, device=device)])
            tours[b] = new_tour[:N]
    return tours
