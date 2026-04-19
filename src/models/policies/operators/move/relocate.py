"""
Relocate local search operator.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_relocate(tours, dist_matrix, max_iterations=200, generator=None):
    """
    Vectorized Relocate operator. Moves a node to a new position.
    Removes a node from its current position and re-inserts it elsewhere if it yields an improvement.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix.
        max_iterations (int): Number of attempts.
        generator (torch.Generator, optional): Random generator.

    Returns:
        torch.Tensor: Updated tours.
    """
    device = tours.device
    B, max_len = tours.shape

    batch_indices = torch.arange(B, device=device).view(B, 1)

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample indices i (to move) and j (insert after)
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i, j = idx[:, 0:1], idx[:, 1:2]

        node_i = torch.gather(tours, 1, i)
        mask = (node_i != 0) & (i != j) & (i != j + 1)
        if not mask.any():
            continue

        # 2. Compute gain
        gain = _compute_relocate_gain(tours, dist_matrix, node_i, i, j, batch_indices)
        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            # 3. Apply moves
            tours = _apply_relocate_move(tours, improved, i, j, max_len, device)

    return tours


def _compute_relocate_gain(tours, dist, node_i, i, j, b_idx):
    """Computes improvement gain for relocating node V_i after node V_j."""
    prev_i = torch.gather(tours, 1, i - 1)
    next_i = torch.gather(tours, 1, i + 1)
    node_j = torch.gather(tours, 1, j)
    next_j = torch.gather(tours, 1, j + 1)

    # Removal cost change
    d_remove = -dist[b_idx, prev_i, node_i] - dist[b_idx, node_i, next_i] + dist[b_idx, prev_i, next_i]
    # Insertion cost change
    d_insert = -dist[b_idx, node_j, next_j] + dist[b_idx, node_j, node_i] + dist[b_idx, node_i, next_j]
    return -(d_remove + d_insert)


def _apply_relocate_move(tours, improved, i, j, max_len, device):
    """Updates tour by moving node at i to position after j."""
    B = tours.shape[0]
    seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
    idx_map = seq_b.clone()

    # Case 1: i < j. Node i moves right. Indices in [i, j) shift left.
    m1 = (i < j) & improved
    shift_left = m1 & (seq_b >= i) & (seq_b < j)
    idx_map[shift_left] = seq_b[shift_left] + 1
    idx_map[m1 & (seq_b == j)] = i.expand(-1, max_len)[m1 & (seq_b == j)]

    # Case 2: j < i. Node i moves left. Indices in (j, i) shift right.
    m2 = (j < i) & improved
    shift_right = m2 & (seq_b > j) & (seq_b < i)
    idx_map[shift_right] = seq_b[shift_right] - 1
    idx_map[m2 & (seq_b == j + 1)] = i.expand(-1, max_len)[m2 & (seq_b == j + 1)]

    return torch.gather(tours, 1, idx_map)
