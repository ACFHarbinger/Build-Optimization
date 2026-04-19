"""
Type III Unstringing Operator (vectorized).
"""

from typing import Optional

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_iii_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Vectorized Type III Unstringing local search across a batch of tours using PyTorch.
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
    if N < 8:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    for _iteration in range(max_iterations):
        improved_any = False
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            valid_indices = torch.where((tour > 0) & (tour < N))[0]
            if len(valid_indices) < 6:
                continue

            best_delta, best_move = _find_best_type_iii_move(tour, dist, valid_indices, sample_size, device, generator)

            if best_move is not None:
                i, k, j, l = best_move
                tours[b] = _apply_type_iii_move(tour, i, k, j, l)
                improved_any = True

        if not improved_any:
            break

    return tours if is_batch else tours.squeeze(0)


def _find_best_type_iii_move(
    tour, dist, valid_indices, sample_size, device, generator: Optional[torch.Generator] = None
):
    """Finds the best Type III unstringing move for a single tour."""
    N = len(tour)
    best_delta = 0.0
    best_move = None
    n_valid = len(valid_indices)

    for i_idx in range(n_valid):
        i = valid_indices[i_idx].item()

        # Sample or enumerate (k, j, l)
        if sample_size > 0:
            n_samples = min(sample_size, n_valid**3)
            k_s = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            j_s = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            l_s = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            # Valid Type III: usually i < k < j < l in circular order
            valid = (k_s > i_idx) & (j_s > k_s) & (l_s > j_s)
            k_s, j_s, l_s = k_s[valid], j_s[valid], l_s[valid]
        else:
            # Exhaustive would be O(N^4) - skip for now or use original logic
            # Let's use a simpler loop for demonstration if exhaustive
            continue

        for ks, js, ls in zip(k_s, j_s, l_s, strict=False):
            k, j, l = valid_indices[ks].item(), valid_indices[js].item(), valid_indices[ls].item()
            delta = _evaluate_type_iii_move(tour, dist, i, k, j, l, N)
            if delta < best_delta - IMPROVEMENT_EPSILON:
                best_delta, best_move = delta, (i, k, j, l)

    return best_delta, best_move


def _evaluate_type_iii_move(tour, dist, i, k, j, l, N):
    """Calculates the delta cost for a specific Type III move."""
    v_ip, v_i, v_in = tour[i - 1 if i > 0 else N - 1].item(), tour[i].item(), tour[(i + 1) % N].item()
    v_k, v_kn = tour[k].item(), tour[(k + 1) % N].item()
    v_j, v_jn = tour[j].item(), tour[(j + 1) % N].item()
    v_l, v_ln = tour[l].item(), tour[(l + 1) % N].item()

    # Pattern III: deleted (vi-1, vi), (vi, vi+1), (vk, vk+1), (vj, vj+1), (vl, vl+1)
    # inserted (vi-1, vk), (vi+1, vj), (vk+1, vl), (vjn, vln)?
    # Standard Pattern III (simplified): 5 edges out, 4 back in (removing node Vi)

    removed = dist[v_ip, v_i] + dist[v_i, v_in] + dist[v_k, v_kn] + dist[v_j, v_jn] + dist[v_l, v_ln]
    inserted = dist[v_ip, v_k] + dist[v_in, v_j] + dist[v_kn, v_l] + dist[v_jn, v_ln]
    return (inserted - removed).item()


def _apply_type_iii_move(tour: torch.Tensor, i: int, k: int, j: int, l: int) -> torch.Tensor:
    """Applies a Type III unstringing move to the tour."""
    n = len(tour)
    tl = tour.tolist()

    i_prev = (i - 1) if i > 0 else (n - 1)
    i_next, k_next, j_next, l_next = (i + 1) % n, (k + 1) % n, (j + 1) % n, (l + 1) % n

    # Segments
    s1 = tl[i_next : k + 1] if i_next <= k else tl[i_next:] + tl[: k + 1]
    s2 = tl[k_next : j + 1] if k_next <= j else tl[k_next:] + tl[: j + 1]
    s3 = tl[j_next : l + 1] if j_next <= l else tl[j_next:] + tl[: l + 1]
    rem = tl[l_next : i_prev + 1] if l_next <= i_prev else tl[l_next:] + tl[: i_prev + 1]

    # Pattern: v_{i-1} -> s1_rev -> s2_rev -> s3_rev -> rem
    new_tour = [tl[i_prev]] + s1[::-1] + s2[::-1] + s3[::-1] + rem
    if 0 in new_tour:
        d_idx = new_tour.index(0)
        new_tour = new_tour[d_idx:] + new_tour[:d_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
