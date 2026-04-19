"""
Dynamic-Programming Route Re-optimisation Operator.

Extracts each route in isolation and solves the Travelling Salesman Problem
(TSP) on its customer nodes exactly using the Held-Karp dynamic programming
algorithm.  The optimal node ordering found by the DP replaces the original
route sequence.

This is a pure intensification operator: the set of visited customers and
their route assignment are never changed.  Route loads and revenue are
therefore guaranteed to be preserved.  Only the *within-route* sequencing
is improved.

Algorithm (Held-Karp):
    State  dp[mask][v] — minimum travel cost to visit exactly the nodes
    in bitmask *mask*, starting from the depot (node 0), ending at node *v*.

    Transition  dp[mask | (1<<j)][j] = min(dp[mask][v] + d[v, j])
                for all j not in mask.

    Base        dp[1<<i][i] = d[0, nodes[i]] for each customer i.

    Answer      min_v( dp[full_mask][v] + d[v, 0] ).

    Complexity  O(2^n · n²) time, O(2^n · n) space.

    Practical limit — the DP is tractable for n ≤ ~20 customers per route.
    Routes with more customers are returned unchanged (set *max_nodes* to
    control the cutoff).

For VRPP problems the CVRP and VRPP variants are identical: revenue depends
only on which customers are visited (unchanged here), so the improvement
objective reduces to pure distance minimisation on each route.

Reference:
    Held, M., & Karp, R. M. (1962). "A dynamic programming approach to
    sequencing problems." Journal of the Society for Industrial and Applied
    Mathematics, 10(1), 196-210.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import dp_route_reopt
    >>> improved = dp_route_reopt(routes, dist_matrix, wastes, capacity)
    >>> improved = dp_route_reopt(routes, dist_matrix, wastes, capacity, max_nodes=15)
"""

import math
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core Held-Karp DP
# ---------------------------------------------------------------------------


def _held_karp(
    nodes: List[int],
    dist_matrix: np.ndarray,
) -> Tuple[float, List[int]]:
    """Solve TSP on *nodes* exactly via Held-Karp DP.

    The route starts and ends at the depot (node index 0 in *dist_matrix*).
    All nodes in *nodes* must be visited exactly once.

    Args:
        nodes: Customer node IDs to sequence optimally (must be non-empty).
        dist_matrix: Full distance matrix; depot at index 0.

    Returns:
        ``(optimal_cost, optimal_route)`` where *optimal_route* is a
        permutation of *nodes* giving the minimum round-trip depot→...→depot
        distance.
    """
    n = len(nodes)
    INF = math.inf

    if n == 1:
        cost = float(dist_matrix[0, nodes[0]] + dist_matrix[nodes[0], 0])
        return cost, list(nodes)

    d = dist_matrix

    # dp[mask][i] = min cost: depot → {nodes in mask} → nodes[i]
    # mask is a bitmask over 0..n-1 indexing into *nodes*
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    # Base case: single-node subsets reached directly from depot
    for i in range(n):
        dp[1 << i][i] = float(d[0, nodes[i]])

    # Fill in increasing mask popcount
    for mask in range(1, 1 << n):
        for last in range(n):
            if not (mask >> last & 1):
                continue
            current_cost = dp[mask][last]
            if current_cost == INF:
                continue
            # Extend to each unvisited node
            for nxt in range(n):
                if mask >> nxt & 1:
                    continue
                new_mask = mask | (1 << nxt)
                new_cost = current_cost + d[nodes[last], nodes[nxt]]
                if new_cost < dp[new_mask][nxt]:
                    dp[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = last

    # Find the best final node to return to depot
    full_mask = (1 << n) - 1
    best_cost = INF
    best_last = 0
    for last in range(n):
        cost = dp[full_mask][last] + d[nodes[last], 0]
        if cost < best_cost:
            best_cost = cost
            best_last = last

    # Reconstruct optimal route via parent backtracking
    path_indices: List[int] = []
    mask = full_mask
    curr = best_last
    while curr != -1:
        path_indices.append(curr)
        prev = parent[mask][curr]
        mask ^= 1 << curr
        curr = prev

    path_indices.reverse()
    optimal_route = [nodes[idx] for idx in path_indices]

    return best_cost, optimal_route


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dp_route_reopt(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    max_nodes: int = 20,
    max_iter: int = 1,
) -> List[List[int]]:
    """Re-optimise each route to its exact TSP optimum via Held-Karp DP.

    For each route with at most *max_nodes* customers the Held-Karp DP is run
    to find the minimum-distance ordering of that route's nodes.  Routes that
    exceed *max_nodes* are returned unchanged (they would require exponential
    memory and time).

    This operator is typically applied once (``max_iter = 1``) immediately
    after construction or after a large ALNS perturbation.  Running it
    repeatedly provides no additional benefit once each route is at its DP
    optimum.

    Route assignments and the set of visited customers are never altered, so
    capacity constraints and revenue are fully preserved.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix of shape ``(N+1, N+1)``, depot at
            index 0.
        wastes: Node demand lookup (unused; included for API consistency).
        capacity: Maximum vehicle load (unused; route assignments are fixed).
        max_nodes: Maximum route length for which the DP is applied.  Routes
            longer than this are skipped.  Recommended: ≤ 20 (at n = 20 the
            DP uses ~20 MB and ~400 M operations).
        max_iter: Number of times to repeat the per-route DP pass.  Defaults
            to 1; additional passes bring no benefit since the DP is exact.

    Returns:
        List[List[int]]: Routes with each eligible route replaced by its
            optimal sequencing.  The input *routes* is never mutated.

    Example:
        >>> improved = dp_route_reopt(routes, dist_matrix, wastes, capacity)
        >>> improved = dp_route_reopt(routes, dist_matrix, wastes, capacity,
        ...                           max_nodes=15)
    """
    working = [list(r) for r in routes if r]
    if not working:
        return working

    for _ in range(max_iter):
        improved_any = False
        for r_idx, route in enumerate(working):
            n = len(route)
            if n < 3 or n > max_nodes:
                # n < 3: trivially optimal; n > max_nodes: skip for tractability
                continue

            original_cost = (
                dist_matrix[0, route[0]]
                + sum(dist_matrix[route[k], route[k + 1]] for k in range(n - 1))
                + dist_matrix[route[-1], 0]
            )

            dp_cost, dp_route = _held_karp(route, dist_matrix)

            if dp_cost < original_cost - 1e-8:
                working[r_idx] = dp_route
                improved_any = True

        if not improved_any:
            break

    return [r for r in working if r]


def dp_route_reopt_profit(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    max_nodes: int = 20,
    max_iter: int = 1,
) -> List[List[int]]:
    """VRPP wrapper for DP route re-optimisation.

    Delegates to :func:`dp_route_reopt`.  Revenue depends only on which
    customers are visited (unchanged here), so the improvement objective
    reduces to distance minimisation — identical to the CVRP variant.

    Args:
        routes: Current plan.
        dist_matrix: Distance matrix (depot at index 0).
        wastes: Node demand lookup (unused).
        capacity: Vehicle capacity (unused).
        R: Revenue per unit waste (unused; revenue unchanged by re-sequencing).
        C: Cost per unit distance (unused; DP minimises raw distance).
        max_nodes: Maximum route length for DP application.
        max_iter: Number of DP passes.

    Returns:
        List[List[int]]: Routes with each eligible route at its TSP optimum.
    """
    return dp_route_reopt(
        routes,
        dist_matrix,
        wastes,
        capacity,
        max_nodes=max_nodes,
        max_iter=max_iter,
    )
