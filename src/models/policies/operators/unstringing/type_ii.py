"""
Type II Unstringing Operator (vectorized).
"""

from typing import Optional

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_ii_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 100,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Vectorized Type II Unstringing local search across a batch of tours using PyTorch.
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
    if N < 7:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    for _iteration in range(max_iterations):
        improved_any = False
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            valid_indices = torch.where((tour > 0) & (tour < N))[0]
            if len(valid_indices) < 5:
                continue

            best_delta, best_move = _find_best_type_ii_move(tour, dist, valid_indices, sample_size, device, generator)

            if best_move is not None:
                i, j, k = best_move
                tours[b] = _apply_type_ii_move(tour, i, j, k)
                improved_any = True

        if not improved_any:
            break

    return tours if is_batch else tours.squeeze(0)


def _find_best_type_ii_move(
    tour, dist, valid_indices, sample_size, device, generator: Optional[torch.Generator] = None
):
    """Finds the best Type II unstringing move for a single tour."""
    N = len(tour)
    best_delta = 0.0
    best_move = None
    n_valid = len(valid_indices)

    for i_idx in range(n_valid):
        i = valid_indices[i_idx].item()

        # Determine (j, k) pairs
        if sample_size > 0:
            n_samples = min(sample_size, (n_valid - 2) * (n_valid - 3) // 2)
            k_samples = torch.randint(i_idx + 2, n_valid, (n_samples,), device=device, generator=generator)
            j_samples = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            # For Type II: i < j < k in circular order usually,
            # but the pattern can vary. Let's stick to the original logic's pairs.
            valid_pairs = k_samples > (j_samples + 1)  # simple constraint
            k_samples, j_samples = k_samples[valid_pairs], j_samples[valid_pairs]
        else:
            k_list, j_list = [], []
            for j_idx in range(n_valid):
                for k_idx in range(j_idx + 2, n_valid):
                    if j_idx == i_idx or k_idx == i_idx:
                        continue

                    k_list.append(k_idx)
                    j_list.append(j_idx)
            if not k_list:
                continue

            k_samples = torch.tensor(k_list, device=device)
            j_samples = torch.tensor(j_list, device=device)

        # Evaluate pairs
        for k_s, j_s in zip(k_samples, j_samples, strict=False):
            k, j = valid_indices[k_s].item(), valid_indices[j_s].item()
            delta = _evaluate_type_ii_move(tour, dist, i, j, k, N)
            if delta < best_delta - IMPROVEMENT_EPSILON:
                best_delta, best_move = delta, (i, j, k)

    return best_delta, best_move


def _evaluate_type_ii_move(tour, dist, i, j, k, N):
    """Calculates the delta cost for a specific Type II move."""
    v_ip, v_i, v_in = tour[i - 1 if i > 0 else N - 1].item(), tour[i].item(), tour[(i + 1) % N].item()
    v_j, v_jn = tour[j].item(), tour[(j + 1) % N].item()
    v_k, v_kn = tour[k].item(), tour[(k + 1) % N].item()

    # Pattern II: v_{i-1}->v_k, v_{j+1}->v_i, v_{i+1}->v_j, v_{k+1}->v_{jn} ?
    removed = dist[v_ip, v_i] + dist[v_i, v_in] + dist[v_j, v_jn] + dist[v_k, v_kn]
    inserted = dist[v_ip, v_k] + dist[v_jn, v_in] + dist[v_j, v_kn]
    return (inserted - removed).item()


def _apply_type_ii_move(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Applies a Type II unstringing move to the tour."""
    n = len(tour)
    tl = tour.tolist()

    i_next, j_next, k_next = (i + 1) % n, (j + 1) % n, (k + 1) % n
    i_prev = (i - 1) if i > 0 else (n - 1)

    # Segments
    seg1 = tl[i_next : j + 1] if i_next <= j else tl[i_next:] + tl[: j + 1]
    seg2 = tl[j_next : k + 1] if j_next <= k else tl[j_next:] + tl[: k + 1]
    rem = tl[k_next : i_prev + 1] if k_next <= i_prev else tl[k_next:] + tl[: i_prev + 1]

    # Pattern: v_{i-1} -> seg2_rev -> seg1_rev -> rem
    new_tour = [tl[i_prev]] + seg2[::-1] + seg1[::-1] + rem
    if 0 in new_tour:
        d_idx = new_tour.index(0)
        new_tour = new_tour[d_idx:] + new_tour[:d_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
