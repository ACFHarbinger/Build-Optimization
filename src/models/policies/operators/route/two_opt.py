"""
2-opt local search operator.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_two_opt(tours, distance_matrix, max_iterations=200, generator=None):
    """
    Vectorized 2-opt local search across a batch of tours using PyTorch.

    The 2-opt algorithm is a classic local search heuristic that iteratively improves
    tours by reversing segments. For each pair of edges (i,i+1) and (j,j+1), it checks
    if swapping them reduces total distance:

    Original: ... -> i -> i+1 -> ... -> j -> j+1 -> ...
    2-opt:    ... -> i -> j -> ... -> i+1 -> j+1 -> ...  (reverse middle segment)

    Args:
        tours: Batch of tours [B, N]
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1]
        max_iterations: Maximum number of improvement iterations (default: 200)
        generator (torch.Generator, optional): Random generator.

    Returns:
        torch.Tensor: Improved tours [B, N]
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
    if N < 4:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    batch_indices = torch.arange(B, device=device).view(B, 1)

    # Pre-generate all possible edge pairs (i, j)
    indices = torch.arange(N, device=device)
    i_orig = indices[1:-2]
    j_orig = indices[2:-1]
    I_grid, J_grid = torch.meshgrid(i_orig, j_orig, indexing="ij")
    pair_mask = J_grid > I_grid
    I_vals, J_vals = I_grid[pair_mask], J_grid[pair_mask]
    K = I_vals.size(0)

    for _ in range(max_iterations):
        # 1. Compute gains for all pairs
        gains = _compute_two_opt_gains(tours, distance_matrix, I_vals, J_vals, batch_indices, B, K)

        # 2. Find best gain
        best_gain, best_idx = torch.max(gains, dim=1)
        improved = best_gain > IMPROVEMENT_EPSILON
        if not improved.any():
            break

        # 3. Apply moves
        tours = _apply_two_opt_moves(tours, improved, I_vals, J_vals, best_idx, B, N, device)

    return tours if is_batch else tours.squeeze(0)


def _compute_two_opt_gains(tours, dist, I_vals, J_vals, b_idx, B, K):
    """Computes gains for all possible 2-opt edge swaps."""
    t_prev_i = tours[:, I_vals - 1]
    t_curr_i = tours[:, I_vals]
    t_curr_j = tours[:, J_vals]
    t_next_j = tours[:, J_vals + 1]

    b_exp = b_idx.expand(B, K)
    d_curr = dist[b_exp, t_prev_i, t_curr_i] + dist[b_exp, t_curr_j, t_next_j]
    d_next = dist[b_exp, t_prev_i, t_curr_j] + dist[b_exp, t_curr_i, t_next_j]
    return d_curr - d_next


def _apply_two_opt_moves(tours, improved, I_vals, J_vals, best_idx, B, N, device):
    """Applies best 2-opt moves using segment reversal via index mapping."""
    target_i = I_vals[best_idx].view(B, 1)
    target_j = J_vals[best_idx].view(B, 1)

    k = torch.arange(N, device=device).view(1, N).expand(B, N)
    idx_map = torch.arange(N, device=device).view(1, N).expand(B, N).clone()

    reversal_mask = (k >= target_i) & (k <= target_j) & improved.view(B, 1)
    rev_idx = target_i + target_j - k
    idx_map[reversal_mask] = rev_idx[reversal_mask]

    return torch.gather(tours, 1, idx_map)
