"""
LKH-3 Objective Module.

Provides the lexicographic (penalty, cost) objective used throughout the
Lin-Kernighan-Helsgaun heuristic and all supporting helper modules.

In LKH-3 (Helsgaun 2017) a solution is evaluated by a two-level criterion:
feasibility first, optimality second.  The *penalty* measures total capacity
violation — the sum of excess demand over all VRP route segments — and is
always minimised before tour cost is considered.  For pure TSP instances the
penalty is always zero and the objective reduces to plain cost minimisation.

**Penalty Evaluation (Route-Level)**:
The penalty is computed at the ROUTE level: each route's total demand is
compared against the vehicle capacity, and any excess is accumulated.
This avoids double-counting that occurs in per-node penalty accumulation.

**Augmented Dummy Depot Encoding (Current Standard)**:
To enable native multi-route optimization, the graph is augmented with
explicit dummy depot nodes at indices [N, N+1, ..., N+M-2] where N is the
original graph size and M is the number of vehicles.

Example (N=5 original nodes, M=3 vehicles):
  - Original nodes: [0, 1, 2, 3, 4]  (0 = depot, 1-4 = customers)
  - Augmented graph: [0, 1, 2, 3, 4, 5, 6]  (5, 6 = dummy depots)
  - Tour: [0, 3, 1, 5, 2, 4, 6, 0]  (split at indices >= 5)
  - Routes: [[3, 1], [2, 4]]

This avoids NumPy negative-indexing bugs and provides O(1) array access.

**Lagrangian Relaxation (Subgradient & α-measures)**:
To drive the search towards high-quality solutions, node penalties (π) are
found via Held-Karp subgradient optimization, maximizing the lower bound
of the Minimum 1-Tree. α-measures are then computed using MST sensitivity
to focus the k-opt search on the most promising edges.

Public API
----------
calculate_penalty(tour, waste, capacity) -> float
    Scan a tour and sum all capacity overloads (supports dummy depots).

get_score(tour, distance_matrix, waste, capacity) -> (penalty, cost)
    Return the full (penalty, cost) pair for a tour with dummy depots.

is_better(p1, c1, p2, c2) -> bool
    Lexicographic dominance check: True iff (p1, c1) strictly beats (p2, c2).

split_tour_at_dummies(tour) -> List[List[int]]
    Extract multi-route representation from a dummy-depot-encoded tour.

solve_subgradient(distance_matrix, max_iterations, ...) -> np.ndarray
    Held-Karp subgradient ascent to optimize node penalties π.

compute_alpha_measures(distance_matrix, pi=None) -> np.ndarray
    Compute α-nearness for every edge using MST sensitivity.

get_candidate_set(distance_matrix, alpha_measures, ...) -> Dict[int, List[int]]
    Build sorted candidate list per node restricted to small α-measures.

Typical usage
-------------
>>> from logic.src.policies.lin_kernighan_helsgaun_three.objective import (
...     get_score, is_better, split_tour_at_dummies, solve_subgradient
... )
>>> # 1. Optimize penalties and compute candidates
>>> pi = solve_subgradient(distance_matrix)
>>> alpha = compute_alpha_measures(distance_matrix, pi)
>>> candidates = get_candidate_set(distance_matrix, alpha)
>>> # 2. Evaluate solution
>>> pen, cost = get_score(tour, dist, demands, capacity)
>>> if is_better(pen, cost, best_pen, best_cost):
...     routes = split_tour_at_dummies(tour)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

# ---------------------------------------------------------------------------
# Dummy Depot Constants & Utilities
# ---------------------------------------------------------------------------

DEPOT_NODE = 0  # Main depot index (always 0)


def is_dummy_depot(node: int, n_original: Optional[int] = None) -> bool:
    """
    Check if a node represents a secondary vehicle (dummy depot).

    In augmented graph mode (VRPP/CVRP), dummy depots are placed after
    original nodes. In legacy mode, they are represented by negative indices.
    """
    if n_original is not None:
        return node >= n_original
    return node < 0


def is_any_depot(node: int, n_original: Optional[int] = None) -> bool:
    """Check if node is either the main depot or an augmented dummy depot."""
    return node == DEPOT_NODE or is_dummy_depot(node, n_original)


def split_tour_at_dummies(tour: List[int], n_original: Optional[int] = None) -> List[List[int]]:
    """
    Split an augmented Hamiltonian circuit into discrete vehicle routes.

    Segments are delimited by any node identified as a depot or dummy depot.
    Resulting routes contain only customer node indices.
    """
    routes: List[List[int]] = []
    current: List[int] = []

    for node in tour:
        if is_any_depot(node, n_original):
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)

    if current:
        routes.append(current)

    return routes


# ---------------------------------------------------------------------------
# Lexicographic Objective (Penalty/Cost)
# ---------------------------------------------------------------------------


def calculate_penalty(
    tour: List[int],
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int] = None,
) -> float:
    """
    Compute total capacity violation over all routes in the tour.

    Penalty calculation is performed at the route level to avoid double-counting.
    A route only contributes to the penalty if its cumulative load exceeds
    the vehicle capacity.
    """
    if waste is None or capacity is None:
        return 0.0

    penalty = 0.0
    current_load = 0.0

    for node in tour:
        if is_any_depot(node, n_original):
            # Evaluate current segment and reset
            penalty += max(0.0, current_load - capacity)
            current_load = 0.0
        else:
            # Accumulate customer demand
            if 0 <= node < len(waste):
                current_load += waste[node]

    return penalty


def get_score(
    tour: List[int],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Evaluate tour under the LKH-3 lexicographic (penalty, cost) objective.

    Penalty represents feasibility (sum of overloads), while cost represents
    optimality (total edge weight). Feasibility is always prioritized.
    """
    n = len(tour) - 1
    c = 0.0
    for i in range(n):
        curr_node = tour[i]
        next_node = tour[i + 1]

        # Legacy handling (mapping negatives to depot)
        if n_original is None:
            if is_dummy_depot(curr_node, None):
                curr_node = DEPOT_NODE
            if is_dummy_depot(next_node, None):
                next_node = DEPOT_NODE

        if 0 <= curr_node < len(distance_matrix) and 0 <= next_node < len(distance_matrix):
            c += distance_matrix[curr_node, next_node]

    pen = calculate_penalty(tour, waste, capacity, n_original)
    return pen, c


def is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """
    Lexicographic dominance check.

    (p1, c1) beats (p2, c2) if p1 < p2 OR (p1 == p2 AND c1 < c2).
    Uses a small epsilon (1e-6) for float stability.
    """
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def penalty_delta(
    old_tour: List[int],
    new_tour: List[int],
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int] = None,
) -> float:
    """Compute ΔP = P(new) - P(old). Used for pre-screening k-opt moves."""
    if waste is None or capacity is None:
        return 0.0
    return calculate_penalty(new_tour, waste, capacity, n_original) - calculate_penalty(
        old_tour, waste, capacity, n_original
    )


# ---------------------------------------------------------------------------
# Held-Karp Penalties (Subgradient Optimization)
# ---------------------------------------------------------------------------


def compute_min_1_tree(distance_matrix: np.ndarray, pi: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute the Minimum 1-Tree for the penalized graph.

    The 1-tree consists of a MST of nodes {1..N-1} plus the two cheapest edges
    connected to the depot (node 0). The sum of node penalties (π) drives
    the MST nodes toward degree 2.
    """
    n = len(distance_matrix)
    if n < 3:
        if n == 2:
            return float(distance_matrix[0, 1] + pi[0] + pi[1]), np.array([1, 1]), np.array([[0, 1]])
        return 0.0, np.zeros(n, dtype=int), np.empty((0, 2), dtype=int)

    sub_dist = distance_matrix[1:, 1:]
    sub_pi = pi[1:]
    D = sub_dist + sub_pi[:, np.newaxis] + sub_pi[np.newaxis, :]

    mst_sparse = minimum_spanning_tree(D)
    mst_edges_coo = mst_sparse.tocoo()

    d0 = distance_matrix[0, 1:] + pi[0] + pi[1:]
    nearest = np.argsort(d0)[:2]
    e1, e2 = nearest[0] + 1, nearest[1] + 1

    total_length = float(mst_sparse.sum() + d0[nearest[0]] + d0[nearest[1]])
    degrees = np.zeros(n, dtype=int)
    degrees[0], degrees[e1], degrees[e2] = 2, 1, 1

    edges = [(0, e1), (0, e2)]
    for u, v in zip(mst_edges_coo.row, mst_edges_coo.col, strict=False):
        u_g, v_g = u + 1, v + 1
        edges.append((u_g, v_g))
        degrees[u_g] += 1
        degrees[v_g] += 1

    return total_length, degrees, np.array(edges)


def solve_subgradient(
    distance_matrix: np.ndarray,
    max_iterations: int = 200,
    n_original: Optional[int] = None,
    initial_pi: Optional[np.ndarray] = None,
    initial_step: Optional[float] = None,
) -> np.ndarray:
    """
    Optimize node penalties π via Held-Karp Lagrangian relaxation.

    The penalties π_i are added to edges incident to node i to 'encourage'
    the MST to resemble a Hamiltonian circuit (all degrees = 2).
    """
    n = len(distance_matrix)
    if n < 3:
        return np.zeros(n)

    pi = initial_pi if initial_pi is not None else np.zeros(n)
    best_lb = -np.inf
    t = initial_step if initial_step is not None else 1.0 / n
    halving_interval = max(1, max_iterations // 5)

    for i in range(max_iterations):
        lb_pi, degrees, _ = compute_min_1_tree(distance_matrix, pi)
        current_lb = lb_pi - 2 * np.sum(pi)
        best_lb = max(best_lb, current_lb)  # type: ignore[assignment]

        G = degrees - 2
        norm_sq = np.sum(G**2)
        if norm_sq == 0:
            break

        step = t * (1.05 * best_lb - current_lb + 1e-4) / norm_sq
        pi += step * G
        pi[0] = 0.0
        if n_original is not None:
            pi[n_original:] = 0.0

        # Held-Karp schedule: halve step every max_iterations // 5 iterations
        if i > 0 and i % halving_interval == 0:
            t *= 0.5

    return pi


# ---------------------------------------------------------------------------
# α-Nearness & Candidate Generation
# ---------------------------------------------------------------------------


def _compute_all_pairs_max_edge(mst_adj: np.ndarray, n: int) -> np.ndarray:
    """O(N²) BFS pass to find heaviest edge on MST paths between all pairs."""
    max_edge = np.zeros((n, n))
    adj: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if mst_adj[i, j] > 0:
                adj[i].append((j, mst_adj[i, j]))
                adj[j].append((i, mst_adj[i, j]))

    for start in range(n):
        visited, queue = np.zeros(n, bool), deque([(start, 0.0)])
        visited[start] = True
        while queue:
            curr, p_max = queue.popleft()
            max_edge[start, curr] = p_max
            for nb, w in adj[curr]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append((nb, max(p_max, w)))
    return max_edge


def compute_alpha_measures(distance_matrix: np.ndarray, pi: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute α-measures using MST sensitivity.

    α(i,j) measures the 'distance' of edge (i,j) from the MST. Edges with
    low α are likely constituents of the optimal tour.
    """
    n = len(distance_matrix)
    pi = pi if pi is not None else np.zeros(n)
    d_pen = distance_matrix + pi[:, np.newaxis] + pi[np.newaxis, :]

    mst_sparse = minimum_spanning_tree(d_pen)
    beta = _compute_all_pairs_max_edge(mst_sparse.toarray(), n)

    alpha = np.maximum(d_pen - beta, 0.0)
    np.fill_diagonal(alpha, 0.0)
    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray,
    alpha_measures: np.ndarray,
    max_candidates: int = 5,
) -> Dict[int, List[int]]:
    """Build per-node candidate lists restricted to small α-measures."""
    n = len(distance_matrix)
    candidates: Dict[int, List[int]] = {}
    for i in range(n):
        indices = sorted(
            [j for j in range(n) if j != i],
            key=lambda j: (alpha_measures[i, j], distance_matrix[i, j]),
        )
        candidates[i] = indices[:max_candidates]
    return candidates
