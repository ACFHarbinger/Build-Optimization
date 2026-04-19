"""
Shaw removal (relatedness-based) operator (vectorized).

The Shaw removal heuristic removes nodes that are similar (related) to a seed node
based on distance, time windows, and waste. This creates clusters of related nodes
that can be efficiently rearranged during repair.
"""

from typing import Optional, Tuple

import torch


def vectorized_shaw_removal(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    n_remove: int,
    wastes: Optional[torch.Tensor] = None,
    time_windows: Optional[torch.Tensor] = None,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
    randomization_factor: float = 2.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized Shaw removal across a batch of tours using PyTorch.

    Shaw removal computes relatedness between nodes based on multiple criteria:
    - Distance: phi * d(i,j) / max_dist
    - Time windows: chi * |T_i - T_j| / max_time
    - Waste: psi * |q_i - q_j| / max_waste

    Nodes with low relatedness scores (high similarity) to already-removed nodes
    are selected for removal, creating clusters of related nodes.

    Algorithm:
    1. Select random seed node for each tour in batch
    2. While removed < n_remove:
        a. Compute relatedness of all remaining nodes to removed set
        b. Rank by relatedness (lower = more related)
        c. Select using randomized power law: idx = (uniform^p) * len(candidates)
        d. Add to removed set
    3. Return tours with selected nodes marked for removal

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        n_remove: Number of nodes to remove from each tour
        wastes: Node wastes [B, N+1] or [N+1] (optional, for waste relatedness)
        time_windows: Time windows [B, N+1, 2] or [N+1, 2] (optional, [earliest, latest])
        phi: Weight for distance component (default: 9.0)
        chi: Weight for time window component (default: 3.0)
        psi: Weight for waste component (default: 2.0)
        randomization_factor: Power for randomized selection (default: 2.0)
            Higher values = more randomness
        generator (Optional[torch.Generator]): PyTorch generator for random number generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Modified tours [B, N] with removed nodes replaced by padding (-1)
            - Removed nodes [B, n_remove] indices of removed nodes

    Note:
        - Tours should include depot as node 0
        - Removed nodes are marked as -1 in returned tours
        - Works with both batched and shared distance/waste matrices
        - Time windows are optional; if not provided, chi is ignored
    """
    # Prepare inputs
    tours, distance_matrix, wastes, time_windows, is_batch = _prepare_shaw_inputs(
        tours, distance_matrix, wastes, time_windows
    )
    device = tours.device

    # Initialize parameters and tracks
    distance_matrix, wastes, time_windows, max_dist, max_waste, max_time = _init_shaw_params(
        tours, distance_matrix, wastes, time_windows
    )
    B, N = tours.shape
    removed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    removed_list = torch.full((B, n_remove), -1, dtype=torch.long, device=device)
    removed_count = torch.zeros(B, dtype=torch.long, device=device)

    valid_mask = tours >= 0
    valid_counts = valid_mask.sum(dim=1)

    # Random seed selection
    _select_seed_nodes(B, valid_mask, valid_counts, removed_mask, removed_list, removed_count, device, generator)

    # Iteratively select related nodes
    for _step in range(1, n_remove):
        relatedness_sum = _calculate_relatedness_batch(
            B,
            N,
            tours,
            distance_matrix,
            wastes,
            time_windows,
            removed_mask,
            removed_count,
            valid_counts,
            max_dist,
            max_waste,
            max_time,
            phi,
            psi,
            chi,
            device,
        )

        relatedness_scores = relatedness_sum
        relatedness_scores[removed_mask] = float("inf")
        relatedness_scores[~valid_mask] = float("inf")

        _select_next_removal_batch(
            B,
            n_remove,
            randomization_factor,
            relatedness_scores,
            removed_mask,
            removed_list,
            removed_count,
            valid_mask,
            valid_counts,
            device,
            generator,
        )

    # Create modified tours with removed nodes marked as -1
    modified_tours = tours.clone()
    for b in range(B):
        modified_tours[b, removed_mask[b]] = -1

    return (
        modified_tours if is_batch else modified_tours.squeeze(0),
        removed_list if is_batch else removed_list.squeeze(0),
    )


def _prepare_shaw_inputs(tours, distance_matrix, wastes, time_windows):
    """Ensures all inputs are batched tensors."""
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)
        if distance_matrix.dim() == 3:
            distance_matrix = distance_matrix.unsqueeze(0)
        if wastes is not None and wastes.dim() == 2:
            wastes = wastes.unsqueeze(0)
        if time_windows is not None and time_windows.dim() == 3:
            time_windows = time_windows.unsqueeze(0)
    return tours, distance_matrix, wastes, time_windows, is_batch


def _select_seed_nodes(B, valid_mask, valid_counts, removed_mask, removed_list, removed_count, device, generator):
    """Selects a random seed node for each tour."""
    for b in range(B):
        if valid_counts[b] > 0:
            valid_indices = torch.where(valid_mask[b])[0]
            seed_idx = valid_indices[torch.randint(len(valid_indices), (1,), device=device, generator=generator)]
            removed_mask[b, seed_idx] = True
            removed_list[b, 0] = seed_idx
            removed_count[b] = 1


def _calculate_relatedness_batch(
    B,
    N,
    tours,
    distance_matrix,
    wastes,
    time_windows,
    removed_mask,
    removed_count,
    valid_counts,
    max_dist,
    max_waste,
    max_time,
    phi,
    psi,
    chi,
    device,
):
    """Calculates relatedness scores for all nodes against removed nodes."""
    relatedness_sum = torch.zeros((B, N), device=device)

    for b in range(B):
        if removed_count[b] >= valid_counts[b]:
            continue

        rem_nodes = tours[b, removed_mask[b]]  # [n_rem]

        # Distance component
        d_rel = distance_matrix[b][tours[b].unsqueeze(1), rem_nodes.unsqueeze(0)] / max_dist
        rel = phi * d_rel

        # Waste component
        if wastes is not None:
            q_rel = torch.abs(wastes[b][tours[b]].unsqueeze(1) - wastes[b][rem_nodes].unsqueeze(0)) / max_waste
            rel += psi * q_rel

        # Time window component
        if time_windows is not None:
            t_rel = (
                torch.abs(time_windows[b][tours[b], 0].unsqueeze(1) - time_windows[b][rem_nodes, 0].unsqueeze(0))
                / max_time
            )
            rel += chi * t_rel

        relatedness_sum[b] = rel.mean(dim=1)

    return relatedness_sum


def _select_next_removal_batch(
    B,
    n_remove,
    randomization_factor,
    relatedness_scores,
    removed_mask,
    removed_list,
    removed_count,
    valid_mask,
    valid_counts,
    device,
    generator,
):
    """Selects the next node to remove using randomized power law."""
    for b in range(B):
        if removed_count[b] >= n_remove or removed_count[b] >= valid_counts[b]:
            continue

        # Get candidates (non-removed, non-padding, finite relatedness)
        candidates_mask = (~removed_mask[b]) & valid_mask[b] & (relatedness_scores[b] < float("inf"))

        if not candidates_mask.any():
            break

        candidate_indices = torch.where(candidates_mask)[0]
        candidate_scores = relatedness_scores[b, candidates_mask]

        # Sort by relatedness (lower = more related)
        sorted_indices = torch.argsort(candidate_scores)
        sorted_candidates = candidate_indices[sorted_indices]

        # Randomized selection using power law
        y = torch.rand(1, device=device, generator=generator).item()
        idx = int((y**randomization_factor) * len(sorted_candidates))
        idx = min(idx, len(sorted_candidates) - 1)

        selected_node_idx = sorted_candidates[idx]

        # Mark as removed
        removed_mask[b, selected_node_idx] = True
        removed_list[b, removed_count[b]] = selected_node_idx
        removed_count[b] += 1


def _init_shaw_params(tours, distance_matrix, wastes, time_windows):
    """Initialize max values for normalization."""
    B, N = tours.shape

    # Ensure distance matrix is batched
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0).expand(B, -1, -1)

    max_dist = distance_matrix.max().item()
    max_dist = max_dist if max_dist > 1e-6 else 1.0

    max_waste = 1.0
    if wastes is not None:
        if wastes.dim() == 1:
            wastes = wastes.unsqueeze(0).expand(B, -1)
        max_waste = wastes.max().item()
        max_waste = max_waste if max_waste > 1e-6 else 1.0

    max_time = 1.0
    if time_windows is not None:
        if time_windows.dim() == 2:
            time_windows = time_windows.unsqueeze(0).expand(B, -1, -1)
        # Max of earliest times
        max_time = time_windows[:, :, 0].max().item()
        max_time = max_time if max_time > 1e-6 else 1.0

    return distance_matrix, wastes, time_windows, max_dist, max_waste, max_time
