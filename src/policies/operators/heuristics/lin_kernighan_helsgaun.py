"""
LKH-3 inspired Heuristic Module.

This module implements a version of the Lin-Kernighan-Helsgaun heuristic inspired by LKH-3.
Key features include Alpha-measure pruning, lexicographical optimization (Penalty, Cost),
and Iterated Local Search with 2-opt and 3-opt moves.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Alpha-measures for edge pruning based on MST.

    alpha(i,j) = c(i,j) - (max edge weight on the unique path in MST between i and j).
    """
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst = mst_sparse.toarray()

    # Alpha approximation: MST edges have alpha=0, others use distance.
    # Refinement: In full LKH, alpha is strictly defined. Here we use this proxy
    # for performance in Python.
    alpha = np.copy(distance_matrix)
    mst_mask = (mst > 0) | (mst.T > 0)
    alpha[mst_mask] = 0

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray, alpha_measures: np.ndarray, max_candidates: int = 5
) -> Dict[int, List[int]]:
    """Generate candidate sets of edges for each node based on Alpha-measures."""
    n = len(distance_matrix)
    candidates = {}
    for i in range(n):
        # Sort by alpha, then by distance
        # We can implement a composite sort key if needed, but alpha is primary.
        indices = np.argsort(alpha_measures[i])
        # Filter self
        valid_indices = [int(idx) for idx in indices if idx != i]
        # Strict limit
        candidates[i] = valid_indices[:max_candidates]
    return candidates


def calculate_penalty(tour: List[int], waste: Optional[np.ndarray], capacity: Optional[float]) -> float:
    """Calculate VRP capacity violation penalty."""
    if waste is None or capacity is None:
        return 0.0

    penalty = 0.0
    current_load = 0.0
    for node in tour:
        if node == 0:
            current_load = 0.0
        else:
            current_load += waste[node]
            if current_load > capacity + 1e-6:
                penalty += current_load - capacity
    return penalty


def get_score(
    tour: List[int],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[float, float]:
    """
    Calculate total penalty and cost for a given tour.

    Returns:
        tuple: (penalty, cost)
    """
    n = len(tour) - 1  # items in tour (excluding duplicate end)
    # Assumes tour is closed [0...0]

    # Cost
    c = 0.0
    for i in range(n):
        c += distance_matrix[tour[i], tour[i + 1]]

    pen = calculate_penalty(tour, waste, capacity)
    return pen, c


def is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """Lexicographical comparison: Penalty first, then Cost."""
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def apply_2opt_move(tour: List[int], i: int, j: int) -> List[int]:
    """
    Apply 2-opt move: reverse segment between i+1 and j.
    Indices refer to positions in the open tour (0 to n-1).
    """
    # Create new tour
    # tour is [0, A, B, ..., 0]
    # We operate on the sequence of nodes.
    # Standard 2-opt on indices i, j implies edge (i, i+1) and (j, j+1) are broken.
    # New edges: (i, j) and (i+1, j+1).
    # Segment reversed: i+1 to j.

    new_tour = tour[:]
    # Python slicing is [start : end] exclusive.
    # Reverse elements from i+1 up to j (inclusive)
    new_tour[i + 1 : j + 1] = new_tour[i + 1 : j + 1][::-1]
    return new_tour


def apply_3opt_move(tour: List[int], i: int, j: int, k: int, case: int) -> List[int]:
    """
    Apply 3-opt move.
    There are multiple ways to reconnect 3 segments.
    Simple case: reverse (i+1..j) and (j+1..k).
    """
    # This roughly corresponds to one of the 3-opt cases (e.g. Type IV US is a 3-opt/4-opt variant).
    # A standard 3-opt move involves removing 3 edges and adding 3.
    # Common reconstruction:
    # Edges removed: (i, i+1), (j, j+1), (k, k+1)
    # Case 0: i->j, i+1->k, j+1->k+1 (reverses i+1..j and j+1..k)

    new_tour = tour[:]

    if case == 0:
        # Reverse i+1 ... j
        new_tour[i + 1 : j + 1] = new_tour[i + 1 : j + 1][::-1]
        # Reverse j+1 ... k
        new_tour[j + 1 : k + 1] = new_tour[j + 1 : k + 1][::-1]

    # Other cases exist, but for this refinement we stick to the most common symmetric 3-opt
    # which is equivalent to swapping segments logic.

    return new_tour


def double_bridge_kick(tour: List[int]) -> List[int]:
    """
    Apply a Double Bridge kick (random 4-opt move).
    Breaks 4 edges and reconnects to create a major perturbation.
    """
    n = len(tour) - 1  # active nodes
    if n < 8:
        return tour  # Too small

    pos = sorted(np.random.choice(range(1, n - 1), 4, replace=False))
    a, b, c, d = pos
    # Segments: [0..a], [a+1..b], [b+1..c], [c+1..d], [d+1..end]
    # Reconnect: [0..a] -> [c+1..d] -> [b+1..c] -> [a+1..b] -> [d+1..end]

    new_tour = tour[: a + 1] + tour[c + 1 : d + 1] + tour[b + 1 : c + 1] + tour[a + 1 : b + 1] + tour[d + 1 :]
    return new_tour


def _initialize_tour(distance_matrix: np.ndarray, initial_tour: Optional[List[int]]) -> List[int]:
    """Initialize tour using nearest neighbor if not provided."""
    n = len(distance_matrix)
    if initial_tour is None:
        # Nearest Neighbor Construction
        curr = 0
        path = [0]
        unvisited = set(range(1, n))
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
            path.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        path.append(0)
        return path
    else:
        curr_tour = initial_tour[:]
        if curr_tour[0] != curr_tour[-1]:
            curr_tour.append(curr_tour[0])
        return curr_tour


def _try_2opt_move(
    curr_tour: List[int],
    i: int,
    t1: int,
    t2: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """Attempt 2-opt moves for edge (t1, t2)."""
    nodes_count = len(curr_tour) - 1
    # Check candidates for t2
    for t3 in candidates[t2]:
        if t3 == t1:
            continue
        if t3 == curr_tour[(i + 2) % nodes_count]:
            continue  # skip next edge

        # 2-opt Attempt
        try:
            j = curr_tour.index(t3)
        except ValueError:
            continue

        # Case A: j > i
        if j <= i + 1:
            continue

        t4 = curr_tour[j + 1]

        # Gain check roughly
        gain = (distance_matrix[t1, t2] + distance_matrix[t3, t4]) - (distance_matrix[t1, t3] + distance_matrix[t2, t4])

        if gain > 1e-6:
            new_tour = apply_2opt_move(curr_tour, i, j)
            p_new, c_new = get_score(new_tour, distance_matrix, waste, capacity)
            return new_tour, p_new, c_new, True, j

    return None, 0.0, 0.0, False, -1


def _try_3opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[Optional[List[int]], float, float, bool]:
    """Attempt 3-opt moves given a failing 2-opt configuration."""
    nodes_count = len(curr_tour) - 1
    for k in range(j + 2, nodes_count):
        t5 = curr_tour[k]
        t6 = curr_tour[k + 1]

        # Check gain
        gain3 = (distance_matrix[t1, t2] + distance_matrix[t3, t4] + distance_matrix[t5, t6]) - (
            distance_matrix[t1, t3] + distance_matrix[t2, t5] + distance_matrix[t4, t6]
        )

        if gain3 > -1e-6:
            new_3opt = apply_3opt_move(curr_tour, i, j, k, 0)
            p3, c3 = get_score(new_3opt, distance_matrix, waste, capacity)
            return new_3opt, p3, c3, True

    return None, 0.0, 0.0, False


def _improve_tour(
    curr_tour: List[int],
    curr_pen: float,
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
) -> Tuple[List[int], float, float, bool]:
    """Run one pass of local search improvement."""
    nodes_count = len(curr_tour) - 1

    for i in range(nodes_count):
        t1 = curr_tour[i]
        t2 = curr_tour[i + 1]

        # 1. Try 2-opt
        # We need to iterate candidates manually to support 3-opt fallback
        # Re-implementing logic from original to support extraction
        for t3 in candidates[t2]:
            # For very small instances, allow all moves to find the best ordering
            if nodes_count >= 5:
                if t3 == t1 or t3 == curr_tour[(i + 2) % nodes_count]:
                    continue
            else:
                if t3 == t2:
                    continue
            try:
                j = curr_tour.index(t3)
            except ValueError:
                continue

            if j <= i + 1:
                continue

            t4 = curr_tour[j + 1]
            gain = (distance_matrix[t1, t2] + distance_matrix[t3, t4]) - (
                distance_matrix[t1, t3] + distance_matrix[t2, t4]
            )

            if gain > -1e-6:
                new_tour = apply_2opt_move(curr_tour, i, j)
                p_new, c_new = get_score(new_tour, distance_matrix, waste, capacity)
                if is_better(p_new, c_new, curr_pen, curr_cost):
                    # print(f"DEBUG LKH: Improved! {curr_tour} -> {new_tour} (p:{curr_pen}->{p_new}, c:{curr_cost}->{c_new})")
                    return new_tour, p_new, c_new, True

            # 2. Try 3-opt if 2-opt didn't improve locally but was a candidate
            # Only do 3-opt for smaller instances or limit breadth
            if len(distance_matrix) < 500:
                res_tour, res_p, res_c, res_imp = _try_3opt_move(
                    curr_tour,
                    i,
                    j,
                    t1,
                    t2,
                    t3,
                    t4,
                    distance_matrix,
                    waste,
                    capacity,
                )
                if res_imp and res_tour is not None and is_better(res_p, res_c, curr_pen, curr_cost):
                    # Explicit narrowing for Pyrefly
                    return res_tour, res_p, res_c, True

    return curr_tour, curr_pen, curr_cost, False


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    waste: Optional[np.ndarray] = None,
    capacity: Optional[float] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float]:
    """
    Solve TSP/VRP using refined Lin-Kernighan heuristics.

    Args:
        distance_matrix: NxN distance matrix.
        initial_tour: Optional initial tour.
        max_iterations: Max improvements/kicks.
        waste: Node weights.
        capacity: Vehicle capacity.

    Returns:
        (best_tour, best_cost)
    """
    n = len(distance_matrix)
    if n < 3:
        tour = list(range(n)) + [0]
        cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(n))
        return tour, float(cost)

    # 1. Initialization
    curr_tour = _initialize_tour(distance_matrix, initial_tour)

    # 2. Candidates
    alpha = compute_alpha_measures(distance_matrix)
    candidates = get_candidate_set(distance_matrix, alpha, max_candidates=5)

    curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity)
    best_tour, best_pen, best_cost = curr_tour[:], curr_pen, curr_cost

    # 3. Main Loop (Iterated Local Search using Kicks)
    for _restart in range(max_iterations):
        # Local Search Loop (2-opt / 3-opt until local optimal)
        while True:
            # delegated to helper
            curr_tour, curr_pen, curr_cost, improved_local = _improve_tour(
                curr_tour,
                curr_pen,
                curr_cost,
                candidates,
                distance_matrix,
                waste,
                capacity,
            )
            if not improved_local:
                break

        # Update Global Best
        if is_better(curr_pen, curr_cost, best_pen, best_cost):
            best_tour, best_pen, best_cost = curr_tour[:], curr_pen, curr_cost

        if recorder is not None:
            recorder.record(restart=_restart, best_cost=best_cost, curr_cost=curr_cost, best_penalty=best_pen)

        # Perturbation (Kick)
        curr_tour = double_bridge_kick(best_tour)
        curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity)

    return best_tour, best_cost
