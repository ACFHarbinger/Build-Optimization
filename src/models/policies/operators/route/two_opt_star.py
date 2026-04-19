"""
2-opt* local search operator (tail swap).
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_two_opt_star(tours, dist_matrix, max_iterations=200, generator=None):
    """
    Vectorized 2-opt* operator. (Tail Swap).
    Exchanges tails of two different routes. Useful for improving solutions by reconnecting routes.

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
        # 1. Sample indices i, j
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i, j = idx[:, 0:1], idx[:, 1:2]

        node_i, node_j = torch.gather(tours, 1, i), torch.gather(tours, 1, j)
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any():
            continue

        # 2. Identify Routes
        end_i, end_j, route_mask = _identify_two_opt_star_routes(tours, i, j, seq, B)
        mask &= route_mask
        if not mask.any():
            continue

        # 3. Compute Gain
        gain = _compute_two_opt_star_gain(tours, dist_matrix, node_i, node_j, i, j, batch_indices)
        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            tours = _apply_two_opt_star_moves(tours, improved, i, j, end_i, end_j, max_len, seq, device)

    return tours


def _identify_two_opt_star_routes(tours, i, j, seq, B):
    """Identifies route boundaries for tail swap."""
    is_zero = tours == 0
    end_i = torch.argmax((is_zero & (seq > i)).float(), dim=1).view(B, 1)
    end_j = torch.argmax((is_zero & (seq > j)).float(), dim=1).view(B, 1)

    valid = (end_i > i) & (end_j > j)
    inter_route = end_i != end_j
    return end_i, end_j, valid & inter_route


def _compute_two_opt_star_gain(tours, dist, u, v, i, j, b_idx):
    """Computes gain for tail swap (broken edges vs new edges)."""
    un = torch.gather(tours, 1, i + 1)
    vn = torch.gather(tours, 1, j + 1)
    d_curr = dist[b_idx, u, un] + dist[b_idx, v, vn]
    d_new = dist[b_idx, u, vn] + dist[b_idx, v, un]
    return d_curr - d_new


def _apply_two_opt_star_moves(tours, improved, i, j, end_i, end_j, max_len, seq, device):
    """Applies tail swap using index mapping for segments."""
    B = tours.shape[0]
    idx_map = seq.clone()

    # R1 before R2
    m_i_lt_j = (end_i <= j) & improved
    if m_i_lt_j.any():
        idx_map = _map_tail_swap(idx_map, m_i_lt_j, i, j, end_i, end_j, B, max_len)

    # R2 before R1
    m_j_lt_i = (end_j <= i) & improved
    if m_j_lt_i.any():
        idx_map = _map_tail_swap(idx_map, m_j_lt_i, j, i, end_j, end_i, B, max_len)

    return torch.gather(tours, 1, idx_map)


def _map_tail_swap(idx_map, mask, i, j, end_i, end_j, B, max_len):
    """Constructs the index mapping for a tail swap where first route < second route."""
    seq_b = torch.arange(max_len, device=idx_map.device).view(1, max_len).expand(B, max_len)

    _len_t1, len_t2 = end_i - (i + 1), end_j - (j + 1)
    len_gap = j - end_i + 1

    # New R1 tail: maps to [j+1, j+len_t2]
    m1 = mask & (seq_b > i) & (seq_b <= i + len_t2)
    idx_map[m1] = (j + 1).view(B, 1).expand(-1, max_len)[m1] + (seq_b[m1] - (i + 1).view(B, 1).expand(-1, max_len)[m1])

    # Gap shift: maps to [end_i, j]
    m_gap = mask & (seq_b > i + len_t2) & (seq_b <= i + len_t2 + len_gap)
    idx_map[m_gap] = (end_i).view(B, 1).expand(-1, max_len)[m_gap] + (
        seq_b[m_gap] - (i + len_t2 + 1).view(B, 1).expand(-1, max_len)[m_gap]
    )

    # New R2 tail: maps to [i+1, end_i-1]
    m2 = mask & (seq_b > i + len_t2 + len_gap) & (seq_b < end_j.view(B, 1))
    idx_map[m2] = (i + 1).view(B, 1).expand(-1, max_len)[m2] + (
        seq_b[m2] - (i + len_t2 + len_gap + 1).view(B, 1).expand(-1, max_len)[m2]
    )

    return idx_map
