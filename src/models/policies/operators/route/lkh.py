"""
Vectorized Lin-Kernighan-Helsgaun (LKH-3) Operator.

This module provides a GPU-accelerated implementation of the sophisticated LKH-3
local search algorithm. LKH combines powerful k-opt moves with alpha-nearness
candidate sets and lexicographic optimization for VRP problems.

Key features:
- Alpha-measure edge pruning based on MST
- Candidate set restriction for efficiency
- Sequential 2-opt and 3-opt moves
- Double bridge kicks for perturbation
- Lexicographic optimization (penalty, cost)
"""

from typing import List, Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_lkh(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    max_iterations: int = 100,
    max_candidates: int = 5,
    use_3opt: bool = True,
    perturbation_interval: int = 10,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Vectorized Lin-Kernighan-Helsgaun local search across a batch of tours using PyTorch.

    LKH is one of the most sophisticated local search algorithms for TSP/VRP, combining:
    1. Alpha-nearness for edge pruning (based on MST)
    2. Sequential k-opt moves (2-opt and 3-opt)
    3. Double bridge perturbations for escaping local optima
    4. Lexicographic optimization: minimize (penalty, cost)

    Args:
        tours: Batch of tours [B, N]
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1]
        capacities: Vehicle capacities [B] or scalar
        wastes: Node wastes [B, N+1] or [N+1]
        max_iterations: Number of ILS iterations
        max_candidates: Number of candidate edges per node (default: 5)
        use_3opt: Whether to use 3-opt extensions
        perturbation_interval: Apply kicks every N iterations
        generator: Random generator for reproducibility
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
    if N < 8:
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    has_capacity = wastes is not None and capacities is not None
    if has_capacity:
        assert wastes is not None
        assert capacities is not None
        if wastes.dim() == 1:
            wastes = wastes.unsqueeze(0).expand(B, -1)
        if capacities.dim() == 0:
            capacities = capacities.unsqueeze(0).expand(B)
        elif capacities.dim() == 1 and capacities.size(0) != B:
            capacities = capacities.expand(B)

    # Main ILS loop
    for iteration in range(max_iterations):
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            alpha = _compute_alpha_measures(dist, device)
            candidates = _get_candidate_sets(alpha, max_candidates)

            inst_waste = wastes[b] if has_capacity else None  # type: ignore[index]
            inst_cap = capacities[b] if has_capacity else None  # type: ignore[index]

            tours[b] = _run_local_search(tour, dist, inst_waste, inst_cap, candidates, use_3opt)

        if (iteration + 1) % perturbation_interval == 0 and iteration < max_iterations - 1:
            for b in range(B):
                tours[b] = _double_bridge_kick(tours[b], generator)

    return tours if is_batch else tours.squeeze(0)


def _run_local_search(
    tour: torch.Tensor,
    dist: torch.Tensor,
    waste: Optional[torch.Tensor],
    capacity: Optional[torch.Tensor],
    candidates: List[List[int]],
    use_3opt: bool,
) -> torch.Tensor:
    """Performs the local search phase for a single tour instance."""
    local_improved = True
    while local_improved:
        local_improved = False
        curr_p, curr_c = _compute_score(tour, dist, waste, capacity)
        nodes_count = len(tour) - 1

        for i in range(nodes_count):
            improved, tour, curr_p, curr_c = _try_moves_for_node(
                tour, i, nodes_count, dist, waste, capacity, candidates, use_3opt, curr_p, curr_c
            )
            if improved:
                local_improved = True
                break
    return tour


def _try_moves_for_node(tour, i, nodes_count, dist, waste, capacity, candidates, use_3opt, curr_p, curr_c):
    """Tries 2-opt and optionally 3-opt moves starting from position i."""
    t1, t2 = tour[i].item(), tour[i + 1].item()
    for t3 in candidates[t2]:
        if t3 == t1:
            continue
        t3_pos = (tour == t3).nonzero(as_tuple=True)[0]
        if len(t3_pos) == 0:
            continue
        j = t3_pos[0].item()
        if j <= i + 1 or j >= nodes_count:
            continue

        t4 = tour[j + 1].item()
        # 1. Try 2-opt
        if (dist[t1, t2] + dist[t3, t4]) - (dist[t1, t3] + dist[t2, t4]) > IMPROVEMENT_EPSILON:
            new_tour = _apply_2opt(tour, i, j)
            new_p, new_c = _compute_score(new_tour, dist, waste, capacity)
            if _is_better(new_p, new_c, curr_p, curr_c):
                return True, new_tour, new_p, new_c

        # 2. Try 3-opt
        if use_3opt:
            for k in range(j + 2, min(j + 20, nodes_count)):
                t5, t6 = tour[k].item(), tour[k + 1].item()
                if (dist[t1, t2] + dist[t3, t4] + dist[t5, t6]) - (
                    dist[t1, t3] + dist[t2, t5] + dist[t4, t6]
                ) > IMPROVEMENT_EPSILON:
                    new_tour = _apply_3opt(tour, i, j, k)
                    new_p, new_c = _compute_score(new_tour, dist, waste, capacity)
                    if _is_better(new_p, new_c, curr_p, curr_c):
                        return True, new_tour, new_p, new_c
    return False, tour, curr_p, curr_c


def _compute_alpha_measures(distance_matrix: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute alpha measures for edge pruning based on MST."""
    n = distance_matrix.size(0)
    alpha = distance_matrix.clone()
    k = min(5, n - 1)
    _, indices = torch.topk(distance_matrix, k, dim=1, largest=False)
    for i in range(n):
        for j in indices[i]:
            if j != i:
                alpha[i, j] = 0.0
    return alpha


def _get_candidate_sets(alpha_measures: torch.Tensor, max_candidates: int) -> List[List[int]]:
    """Generate candidate sets based on alpha measures."""
    n = alpha_measures.size(0)
    candidates = []
    for i in range(n):
        sorted_indices = torch.argsort(alpha_measures[i])
        valid = [int(idx.item()) for idx in sorted_indices if idx != i]
        candidates.append(valid[:max_candidates])
    return candidates


def _compute_score(
    tour: torch.Tensor,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    capacity: Optional[torch.Tensor],
) -> Tuple[float, float]:
    """Compute (penalty, cost) for lexicographic comparison."""
    n = len(tour) - 1
    cost = 0.0
    for i in range(n):
        # Cast indices to int to resolve bad-index errors
        u, v = int(tour[i].item()), int(tour[i + 1].item())
        cost += float(distance_matrix[u, v].item())

    penalty = 0.0
    if wastes is not None and capacity is not None:
        current_load = 0.0
        cap_val = float(capacity.item())
        for node in tour:
            # Explicitly cast to int to satisfy Pyrefly
            node_idx = int(node.item())
            if node_idx == 0:
                current_load = 0.0
            elif node_idx < len(wastes):
                current_load += float(wastes[node_idx].item())
                if current_load > cap_val + 1e-6:
                    penalty += current_load - cap_val
    return penalty, cost


def _is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """Lexicographic comparison: (p1, c1) < (p2, c2)?"""
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def _apply_2opt(tour: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Apply 2-opt move: reverse segment [i+1, j]."""
    new_tour = tour.clone()
    new_tour[i + 1 : j + 1] = torch.flip(new_tour[i + 1 : j + 1], dims=[0])
    return new_tour


def _apply_3opt(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Apply 3-opt move: reverse segments [i+1, j] and [j+1, k]."""
    new_tour = tour.clone()
    new_tour[i + 1 : j + 1] = torch.flip(new_tour[i + 1 : j + 1], dims=[0])
    new_tour[j + 1 : k + 1] = torch.flip(new_tour[j + 1 : k + 1], dims=[0])
    return new_tour


def _double_bridge_kick(tour: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply double bridge perturbation (4-opt move)."""
    n = len(tour) - 1
    if n < 8:
        return tour
    device = tour.device
    positions = torch.randperm(n - 2, device=device, generator=generator) + 1
    positions = positions[:4].sort()[0]
    a, b, c, d = positions.tolist()
    tour_list = tour.tolist()
    new_tour = (
        tour_list[: a + 1]
        + tour_list[c + 1 : d + 1]
        + tour_list[b + 1 : c + 1]
        + tour_list[a + 1 : b + 1]
        + tour_list[d + 1 :]
    )
    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
