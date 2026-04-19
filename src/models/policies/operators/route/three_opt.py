"""
3-opt local search operator.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_three_opt(tours, dist_matrix, max_iterations=100, generator=None):
    """
    Vectorized 3-opt local search using sampling for high efficiency.
    Evaluates multiple reconnection ways for 3-edge removals in parallel.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix (B, N, N) or compatible.
        max_iterations (int): Number of sampling iterations.
        generator (torch.Generator, optional): Random generator.

    Returns:
        torch.Tensor: Updated tours.
    """
    device = tours.device
    B, max_len = tours.shape

    if max_len < 6:
        return tours

    batch_indices = torch.arange(B, device=device).view(B, 1)

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample 3 indices i < j < k
        idx = torch.sort(torch.randint(1, max_len - 1, (B, 3), device=device, generator=generator), dim=1).values
        i, j, k = idx[:, 0:1], idx[:, 1:2], idx[:, 2:3]

        mask = (torch.gather(tours, 1, i) != 0) & (torch.gather(tours, 1, j) != 0) & (torch.gather(tours, 1, k) != 0)
        mask &= (j > i + 1) & (k > j + 1)
        if not mask.any():
            continue

        # 2. Compute gains
        best_gain, best_case = _compute_three_opt_gains(tours, dist_matrix, i, j, k, batch_indices)

        improved = (best_gain > IMPROVEMENT_EPSILON) & mask.squeeze(1)
        if improved.any():
            tours = _apply_three_opt_moves(tours, improved, best_case, i, j, k, max_len, device)

    return tours


def _compute_three_opt_gains(tours, dist, i, j, k, b_idx):
    """Computes gains for cases 4-7 of 3-opt."""
    u, un = torch.gather(tours, 1, i), torch.gather(tours, 1, i + 1)
    v, vn = torch.gather(tours, 1, j), torch.gather(tours, 1, j + 1)
    w, wn = torch.gather(tours, 1, k), torch.gather(tours, 1, k + 1)

    d_base = dist[b_idx, u, un] + dist[b_idx, v, vn] + dist[b_idx, w, wn]

    g4 = d_base - (dist[b_idx, u, v] + dist[b_idx, un, w] + dist[b_idx, vn, wn])
    g5 = d_base - (dist[b_idx, u, vn] + dist[b_idx, w, un] + dist[b_idx, v, wn])
    g6 = d_base - (dist[b_idx, u, vn] + dist[b_idx, w, v] + dist[b_idx, un, wn])
    g7 = d_base - (dist[b_idx, u, w] + dist[b_idx, vn, un] + dist[b_idx, v, wn])

    return torch.max(torch.cat([g4, g5, g6, g7], dim=1), dim=1)


def _apply_three_opt_moves(tours, improved, best_case, i, j, k, max_len, device):
    """Applies improvements for cases 4-7 using index mapping."""
    B = tours.shape[0]
    seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
    idx_map = seq_b.clone()

    # Common segment lengths
    len3 = k - j

    # Case 4 (idx 0): S1, S2^R, S3^R, S4
    m0 = (best_case == 0) & improved
    idx_map = torch.where(m0.view(B, 1) & (seq_b > i) & (seq_b <= j), i + j + 1 - seq_b, idx_map)
    idx_map = torch.where(m0.view(B, 1) & (seq_b > j) & (seq_b <= k), j + k + 1 - seq_b, idx_map)

    # Case 5 (idx 1): S1, S3, S2, S4
    m1 = (best_case == 1) & improved
    idx_map = torch.where(m1.view(B, 1) & (seq_b > i) & (seq_b <= i + len3), seq_b + (j - i), idx_map)
    idx_map = torch.where(m1.view(B, 1) & (seq_b > i + len3) & (seq_b <= k), seq_b - (k - j), idx_map)

    # Case 6 (idx 2): S1, S3, S2^R, S4
    m2 = (best_case == 2) & improved
    idx_map = torch.where(m2.view(B, 1) & (seq_b > i) & (seq_b <= i + len3), seq_b + (j - i), idx_map)
    idx_map = torch.where(m2.view(B, 1) & (seq_b > i + len3) & (seq_b <= k), j - (seq_b - (i + len3 + 1)), idx_map)

    # Case 7 (idx 3): S1, S3^R, S2, S4
    m3 = (best_case == 3) & improved
    idx_map = torch.where(m3.view(B, 1) & (seq_b > i) & (seq_b <= i + len3), k - (seq_b - (i + 1)), idx_map)
    idx_map = torch.where(m3.view(B, 1) & (seq_b > i + len3) & (seq_b <= k), seq_b - (k - j), idx_map)

    return torch.gather(tours, 1, idx_map)
