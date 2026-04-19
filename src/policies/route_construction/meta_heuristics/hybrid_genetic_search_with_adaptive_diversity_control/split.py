"""
Split algorithm for decoding giant tours into valid vehicle routes.
Supports both Unconstrained Fleet (Prins 2004) and Constrained Fleet (Resource CSP).
"""

from typing import List, Tuple

import numpy as np


def compute_daily_loads(
    patterns: np.ndarray, base_wastes: np.ndarray, daily_increments: np.ndarray, T: int
) -> np.ndarray:
    """
    Computes the load q_{i,t} for each node i on day t, assuming the pattern P.

    Args:
        patterns: Shape (N,). Integer where bits represent visit days.
        base_wastes: Shape (N,). Inventory at day 0.
        daily_increments: Shape (T, N). Incremental waste generated per day 1..T.
        T: Horizon length.

    Returns:
        np.ndarray of shape (T, N) containing the physical load picked up if visited.
        If node i is not visited on day t, the value is 0.
    """
    N = patterns.shape[0]
    loads = np.zeros((T, N))

    for i in range(1, N):  # Skip depot (0)
        p_i = patterns[i]
        current_load = base_wastes[i]
        for t in range(T):
            current_load += daily_increments[t, i]
            if (p_i >> t) & 1:
                loads[t, i] = current_load
                current_load = 0.0  # reset after visit
    return loads


def split_day(
    giant_tour: np.ndarray, loads: np.ndarray, distance_matrix: np.ndarray, capacity: float, n_vehicles: int
) -> Tuple[List[List[int]], float, float]:
    """
    Decodes a generic daily giant tour into route sequences.

    Args:
        giant_tour: 1D array of active nodes.
        loads: 1D array of loads for all nodes (shape N). We look up loads[giant_tour[i]].
        distance_matrix: 2D array distance.
        capacity: Max capacity per route constraint (soft for fitness).
        n_vehicles: Hard topological limit on max vehicles (0 = unlimited).

    Returns:
        (Decoded routes, Day Cost, Capacity Violation Sum)
    """
    if len(giant_tour) == 0:
        return [], 0.0, 0.0

    if n_vehicles == 0:
        return _split_unconstrained(giant_tour, loads, distance_matrix, capacity)
    else:
        return _split_constrained(giant_tour, loads, distance_matrix, capacity, n_vehicles)


def _split_unconstrained(
    giant_tour: np.ndarray, loads: np.ndarray, dist: np.ndarray, capacity: float
) -> Tuple[List[List[int]], float, float]:
    """
    O(N) Bellman Split using Prins (2004) unconstrained shortest path.
    """
    n = len(giant_tour)

    # Standard forward iteration
    V = np.full(n + 1, float("inf"))
    V[0] = 0.0
    P = np.zeros(n + 1, dtype=int)
    for i in range(n):
        if V[i] == float("inf"):
            continue

        load = 0.0
        cost = 0.0

        for j in range(i + 1, n + 1):
            node_j = giant_tour[j - 1]
            load += loads[node_j]

            # Transition cost
            if j == i + 1:
                cost = dist[0, node_j] + dist[node_j, 0]
            else:
                prev_node = giant_tour[j - 2]
                cost = cost - dist[prev_node, 0] + dist[prev_node, node_j] + dist[node_j, 0]

            # If load > capacity, normally Split allows it but penalizes.
            # In HGS-ADC, Delta Q penalty is separated from routing cost.
            # But the DAG needs to minimize Total Cost + w_Q * Delta Q?
            # Or just Bellman with infinite capacity. The prompt says:
            # "Track capacity violations \Delta_Q per route."
            # So the Split *does* evaluate violations. For simplicity, cost metric
            # here is standard cost. Wait, split must minimize expected fitness.
            # We'll penalize violations heavily during Split to encourage feasible splits
            # or use a fixed w_Q if available. Assuming w_Q = 100 for Split construction internally.

            current_violation = max(0.0, load - capacity)
            w_q = 100.0  # Heuristic weight for split DAG

            total_arc_cost = V[i] + cost + w_q * current_violation

            if total_arc_cost < V[j]:
                V[j] = total_arc_cost
                P[j] = i

    # Reconstruct
    routes = []
    curr = n
    total_cost = 0.0
    total_violation = 0.0

    while curr > 0:
        start = P[curr]
        route = giant_tour[start:curr].tolist()
        routes.append(route)

        # Calculate true route metrics
        r_cost = dist[0, route[0]]
        r_load = 0.0
        for idx in range(len(route) - 1):
            r_cost += dist[route[idx], route[idx + 1]]
            r_load += loads[route[idx]]
        r_load += loads[route[-1]]
        r_cost += dist[route[-1], 0]

        total_cost += r_cost
        total_violation += max(0.0, r_load - capacity)

        curr = start

    routes.reverse()
    return routes, total_cost, total_violation


def _split_constrained(
    giant_tour: np.ndarray, loads: np.ndarray, dist: np.ndarray, capacity: float, K: int
) -> Tuple[List[List[int]], float, float]:
    """
    O(K * N) Resource-Constrained Shortest Path for max K vehicles.
    """
    n = len(giant_tour)

    # V[k][i] = min cost of routing first i nodes using exactly k vehicles
    V = np.full((K + 1, n + 1), float("inf"))
    V[0][0] = 0.0

    # P[k][i] stores the starting index for the k-th route
    P = np.zeros((K + 1, n + 1), dtype=int)

    w_q = 100.0

    for k in range(1, K + 1):
        for i in range(n):
            if V[k - 1][i] == float("inf"):
                continue

            load = 0.0
            cost = 0.0

            for j in range(i + 1, n + 1):
                node_j = giant_tour[j - 1]
                load += loads[node_j]

                if j == i + 1:
                    cost = dist[0, node_j] + dist[node_j, 0]
                else:
                    prev_node = giant_tour[j - 2]
                    cost = cost - dist[prev_node, 0] + dist[prev_node, node_j] + dist[node_j, 0]

                current_violation = max(0.0, load - capacity)
                total_arc_cost = V[k - 1][i] + cost + w_q * current_violation

                if total_arc_cost < V[k][j]:
                    V[k][j] = total_arc_cost
                    P[k][j] = i

    # Find best K' <= K that covers all nodes
    best_k = -1
    best_k_cost = float("inf")

    for k in range(1, K + 1):
        if V[k][n] < best_k_cost:
            best_k_cost = V[k][n]
            best_k = k

    if best_k == -1:
        # Infeasible to route within K vehicles topologically (should not happen if Split allows infinite load, but just in case)
        # We fallback to a generic split and heavily penalize
        r, c, v = _split_unconstrained(giant_tour, loads, dist, capacity)
        # Add massive penalty for fleet violation
        return r, c + 1e9, v

    routes = []
    curr = n
    curr_k = best_k

    total_cost = 0.0
    total_violation = 0.0

    while curr > 0 and curr_k > 0:
        start = P[curr_k][curr]
        route = giant_tour[start:curr].tolist()
        routes.append(route)

        r_cost = dist[0, route[0]]
        r_load = 0.0
        for idx in range(len(route) - 1):
            r_cost += dist[route[idx], route[idx + 1]]
            r_load += loads[route[idx]]
        r_load += loads[route[-1]]
        r_cost += dist[route[-1], 0]

        total_cost += r_cost
        total_violation += max(0.0, r_load - capacity)

        curr = start
        curr_k -= 1

    routes.reverse()
    return routes, total_cost, total_violation
