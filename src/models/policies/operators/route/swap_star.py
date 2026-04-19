"""
Swap* local search operator (inter-route swap with re-insertion).
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_swap_star(tours, dist_matrix, max_iterations=100, generator=None):
    """
    Vectorized Swap* operator.
    Exchanges u and v between different routes, re-inserting them at best positions.
    This is a more powerful move that combines swapping nodes between routes with re-optimizing their insertion points.

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
    seq = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample u (i) and v (j)
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i, j = idx[:, 0:1], idx[:, 1:2]

        node_i, node_j = torch.gather(tours, 1, i), torch.gather(tours, 1, j)
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any():
            continue

        # 2. Identify Routes
        start_i, end_i, start_j, end_j, route_mask = _identify_routes(tours, i, j, seq, B)
        mask &= route_mask
        if not mask.any():
            continue

        # 3. Compute Gains and Find Best Insertions
        total_gain, best_ins_u, best_ins_v = _compute_swap_star_gains(
            tours,
            dist_matrix,
            node_i,
            node_j,
            i,
            j,
            start_i,
            end_i,
            start_j,
            end_j,
            batch_indices,
            max_len,
            seq,
            device,
        )

        improved = (total_gain > IMPROVEMENT_EPSILON) & mask
        if improved.any():
            tours = _apply_swap_star_moves(tours, improved, i, j, best_ins_u, best_ins_v, max_len, device)

    return tours


def _identify_routes(tours, i, j, seq, B):
    """Identifies route boundaries for nodes i and j."""
    is_zero = tours == 0

    start_i = torch.max(torch.where(is_zero & (seq < i), seq, -1), dim=1)[0].view(B, 1)
    end_i = torch.argmax((is_zero & (seq > i)).float(), dim=1).view(B, 1)

    start_j = torch.max(torch.where(is_zero & (seq < j), seq, -1), dim=1)[0].view(B, 1)
    end_j = torch.argmax((is_zero & (seq > j)).float(), dim=1).view(B, 1)

    valid = (end_i > i) & (start_i < i) & (end_j > j) & (start_j < j) & (start_i >= 0) & (start_j >= 0)
    inter_route = start_i != start_j
    return start_i, end_i, start_j, end_j, valid & inter_route


def _compute_swap_star_gains(
    tours, dist, node_i, node_j, i, j, start_i, end_i, start_j, end_j, b_idx, max_len, seq, device
):
    """Computes gains and identifies best re-insertion points."""
    # Removal gains
    gain_i = (
        dist[b_idx, torch.gather(tours, 1, i - 1), node_i]
        + dist[b_idx, node_i, torch.gather(tours, 1, i + 1)]
        - dist[b_idx, torch.gather(tours, 1, i - 1), torch.gather(tours, 1, i + 1)]
    )
    gain_j = (
        dist[b_idx, torch.gather(tours, 1, j - 1), node_j]
        + dist[b_idx, node_j, torch.gather(tours, 1, j + 1)]
        - dist[b_idx, torch.gather(tours, 1, j - 1), torch.gather(tours, 1, j + 1)]
    )

    # Best insertions
    next_nodes = torch.roll(tours, shifts=-1, dims=1)
    b_rows = b_idx.expand(-1, max_len)

    # u into J
    cost_u = (
        dist[b_rows, tours, node_i.expand(-1, max_len)]
        + dist[b_rows, node_i.expand(-1, max_len), next_nodes]
        - dist[b_rows, tours, next_nodes]
    )
    mask_j = (seq >= start_j) & (seq < end_j) & (seq != j) & (seq != j - 1)
    val_u, idx_u = torch.min(torch.where(mask_j, cost_u, torch.tensor(float("inf"), device=device)), dim=1)

    # v into I
    cost_v = (
        dist[b_rows, tours, node_j.expand(-1, max_len)]
        + dist[b_rows, node_j.expand(-1, max_len), next_nodes]
        - dist[b_rows, tours, next_nodes]
    )
    mask_i = (seq >= start_i) & (seq < end_i) & (seq != i) & (seq != i - 1)
    val_v, idx_v = torch.min(torch.where(mask_i, cost_v, torch.tensor(float("inf"), device=device)), dim=1)

    return gain_i + gain_j - val_u.view(-1, 1) - val_v.view(-1, 1), idx_u.view(-1, 1), idx_v.view(-1, 1)


def _apply_swap_star_moves(tours, improved, i, j, ins_u, ins_v, max_len, device):
    """Applies Swap* moves using priority-based sorting."""
    seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(tours.shape[0], max_len).float()
    weights = seq_b * 10.0

    new_weights = weights.clone()
    new_weights.scatter_(1, i, ins_u.float() * 10.0 + 5.0)
    new_weights.scatter_(1, j, ins_v.float() * 10.0 + 5.1)

    weights = torch.where(improved, new_weights, weights)
    return torch.gather(tours, 1, torch.argsort(weights, dim=1))
