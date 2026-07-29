"""
Greedy insertion repair operator (vectorized).
"""

from typing import Optional

import torch
from torch import Tensor


def vectorized_greedy_insertion(
    tours: Tensor,
    removed_nodes: Tensor,
    dist_matrix: Tensor,
    wastes: Optional[Tensor] = None,
    capacity: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Vectorized greedy insertion.
    Sequentially inserts each node from removed_nodes into the best position in tours.

    Args:
        tours: (B, N_curr)
        removed_nodes: (B, N_rem)
        dist_matrix: (B, N_all, N_all)
        wastes: (N_all) or (B, N_all)
        capacity: float
        generator (Optional[torch.Generator]): PyTorch generator for random number generation.

    Returns:
        tours: (B, N_curr + N_rem)
    """
    B, N_curr = tours.shape
    N_rem = removed_nodes.shape[1]

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)

    for i in range(N_rem):
        node_to_insert = removed_nodes[:, i : i + 1]

        # 1. Compute costs and find best position
        costs = _compute_greedy_insertion_costs(tours, node_to_insert, dist_matrix)
        _, best_indices = torch.min(costs, dim=1)
        insert_pos = best_indices.view(B, 1)

        # 2. Insert node
        tours = _apply_insertion(tours, node_to_insert, insert_pos)
        N_curr += 1

    return tours


def _compute_greedy_insertion_costs(tours, node, dist):
    """Computes insertion costs for a single node at all positions in the tour."""
    B, N_curr = tours.shape
    device = tours.device

    t_prev = tours
    t_next = torch.roll(tours, -1, dims=1)
    node_exp = node.expand(-1, N_curr)

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N_curr)
    d_pn = dist[b_idx, t_prev, node_exp]
    d_nn = dist[b_idx, node_exp, t_next]
    d_pn_exist = dist[b_idx, t_prev, t_next]

    return d_pn + d_nn - d_pn_exist


def _apply_insertion(tours, node, pos):
    """Inserts a node into the tour at the specified position."""
    B, N_curr = tours.shape
    device = tours.device

    new_tours = torch.zeros((B, N_curr + 1), dtype=tours.dtype, device=device)
    seq = torch.arange(N_curr, device=device).unsqueeze(0).expand(B, N_curr)

    mask_left = seq <= pos
    write_idx = torch.where(mask_left, seq, seq + 1)

    new_tours.scatter_(1, write_idx, tours)
    new_tours.scatter_(1, pos + 1, node)
    return new_tours
