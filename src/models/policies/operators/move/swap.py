"""
Swap local search operator.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_swap(tours, dist_matrix, max_iterations=200, generator=None):
    """
    Vectorized Swap operator.
    Exchanges two nodes within the same route for multiple batch items simultaneously.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix.
        max_iterations (int): Number of random swap attempts.
        generator (torch.Generator, optional): Random generator.

    Returns:
        torch.Tensor: Updated tours.
    """
    device = tours.device
    B, max_len = tours.shape

    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

    # Handle dist matrix expansion
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    batch_indices = torch.arange(B, device=device).view(B, 1)

    for _ in range(max_iterations):
        # 1. Sample indices i, j
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i = torch.min(idx, dim=1)[0].view(B, 1)
        j = torch.max(idx, dim=1)[0].view(B, 1)

        node_i, node_j = torch.gather(tours, 1, i), torch.gather(tours, 1, j)
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any():
            continue

        # 2. Compute gain
        gain = _compute_swap_gain(tours, dist_matrix, node_i, node_j, i, j, batch_indices)
        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            # 3. Apply moves
            tours = _apply_swap_moves(tours, improved, i, j)

    return tours if is_batch else tours.squeeze(0)


def _compute_swap_gain(tours, dist, node_i, node_j, i, j, b_idx):
    """Computes improvement gain for swapping node V_i and V_j."""
    p_i, n_i = torch.gather(tours, 1, i - 1), torch.gather(tours, 1, i + 1)
    p_j, n_j = torch.gather(tours, 1, j - 1), torch.gather(tours, 1, j + 1)

    # Current cost
    d_curr = dist[b_idx, p_i, node_i] + dist[b_idx, node_i, n_i]
    d_curr += dist[b_idx, p_j, node_j] + dist[b_idx, node_j, n_j]

    # Special case: adjacent nodes
    # If j == i + 1, then next_i == node_j and prev_j == node_i
    # We need to be careful with double counting the edge between them.

    # New cost
    d_new = dist[b_idx, p_i, node_j] + dist[b_idx, node_j, n_i]
    d_new += dist[b_idx, p_j, node_i] + dist[b_idx, node_i, n_j]

    return d_curr - d_new


def _apply_swap_moves(tours, improved, i, j):
    """Applies swap moves to improved instances in the batch."""
    batch_mask = improved.squeeze(1)
    if batch_mask.any():
        sub_tours = tours[batch_mask]
        sub_i, sub_j = i[batch_mask], j[batch_mask]

        val_i = torch.gather(sub_tours, 1, sub_i)
        val_j = torch.gather(sub_tours, 1, sub_j)

        sub_tours.scatter_(1, sub_i, val_j)
        sub_tours.scatter_(1, sub_j, val_i)
        tours[batch_mask] = sub_tours
    return tours
