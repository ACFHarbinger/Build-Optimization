"""
Regret-K insertion repair operator (vectorized).
"""

from typing import Optional

import torch
from torch import Tensor


def vectorized_regret_k_insertion(
    tours: Tensor, removed_nodes: Tensor, dist_matrix: Tensor, k: int = 2, generator: Optional[torch.Generator] = None
) -> Tensor:
    """
    Vectorized Regret-K insertion.
    Inserts removed nodes into best positions based on the regret criterion.
    """
    B, _ = tours.shape
    device = tours.device
    N_rem = removed_nodes.shape[1]

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)

    pending_mask = torch.ones((B, N_rem), dtype=torch.bool, device=device)

    for _ in range(N_rem):
        # 1. Compute costs and regrets
        costs = _compute_insertion_costs(tours, removed_nodes, dist_matrix)
        regret, top_pos = _compute_regrets(costs, k, pending_mask)

        # 2. Select node with Max Regret
        max_regret_val, node_idx_in_rem = torch.max(regret, dim=1)

        # 3. Apply Insertion
        node_to_insert = torch.gather(removed_nodes, 1, node_idx_in_rem.unsqueeze(1))
        insert_pos = torch.gather(top_pos, 1, node_idx_in_rem.unsqueeze(1))

        tours = _apply_insertion(tours, node_to_insert, insert_pos)
        pending_mask.scatter_(1, node_idx_in_rem.unsqueeze(1), False)

        if not pending_mask.any():
            break

    return tours


def _compute_insertion_costs(tours, removed_nodes, dist):
    """Computes insertion costs for all pending nodes at all positions."""
    B, N_curr = tours.shape
    B_rem, N_rem = removed_nodes.shape
    device = tours.device

    t_prev = tours.unsqueeze(1).expand(-1, N_rem, -1)
    t_next = torch.roll(tours, -1, dims=1).unsqueeze(1).expand(-1, N_rem, -1)
    nodes = removed_nodes.unsqueeze(2).expand(-1, -1, N_curr)

    b_idx = torch.arange(B, device=device).view(B, 1, 1)
    d_pn = dist[b_idx, t_prev, nodes]
    d_nn = dist[b_idx, nodes, t_next]
    d_pn_exist = dist[b_idx, t_prev, t_next]

    return d_pn + d_nn - d_pn_exist


def _compute_regrets(costs, k, pending_mask):
    """Calculates regret based on top-k insertion costs."""
    topk_vals, topk_indices = torch.topk(costs, k=k, dim=2, largest=False)

    best_costs = topk_vals[:, :, 0]
    kth_costs = topk_vals[:, :, -1]

    regret = kth_costs - best_costs
    regret[~pending_mask] = -float("inf")
    return regret, topk_indices[:, :, 0]


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
