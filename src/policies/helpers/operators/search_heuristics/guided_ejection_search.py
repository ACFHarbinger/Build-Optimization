"""
Guided Ejection Search (GES) Operator Module.

This module implements the Guided Ejection Search heuristic, which ejects
an entire route and reinserts its nodes greedily into the remaining routes.
"""

import random
from typing import Dict, List

import numpy as np


def apply_ges(  # noqa: C901
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    rng: random.Random,
    max_depth: int = 3,
) -> List[List[int]]:
    """
    Apply Guided Ejection Search (GES) with ejection chains.

    Unlike standard ruin-operators, GES tunnels through infeasible regions by
    allowing temporary capacity violations, followed by forced ejections to
    restore feasibility.

    Args:
        routes: Current feasible routes.
        dist_matrix: NxN distance matrix.
        wastes: Dictionary of node wastes.
        capacity: Maximum vehicle capacity.
        rng: Random number generator.
        max_depth: Maximum recursion depth for ejection chains.

    Returns:
        List[List[int]]: Modified feasible routes.
    """
    if not routes:
        return routes

    original_routes = [r[:] for r in routes]
    new_routes = [r[:] for r in routes if r]

    if not new_routes:
        return routes

    # 1. Initial Ejection: Select a random route to destroy
    ejected_idx = rng.randrange(len(new_routes))
    pool = new_routes.pop(ejected_idx)

    depth = 0
    while pool and depth < max_depth:
        # 2. Reinsertion: Attempt to insert all nodes in the pool
        # We allow temporary capacity violations during this phase
        while pool:
            node = pool.pop(0)
            best_cost = float("inf")
            # Find best insertion position across all routes (ignoring capacity)
            best_route_idx = -1
            for r_idx, route in enumerate(new_routes):
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos

            if best_route_idx != -1:
                new_routes[best_route_idx].insert(best_pos, node)
            else:
                new_routes.append([node])

        # 3. Feasibility Restoral: Check for violations and eject excess nodes
        violations = []
        for r_idx, route in enumerate(new_routes):
            load = sum(wastes.get(n, 0.0) for n in route)
            if load > capacity:
                violations.append(r_idx)

        if not violations:
            return [r for r in new_routes if r]

        # For each violating route, eject nodes until feasible
        for r_idx in violations:
            route = new_routes[r_idx]
            while sum(wastes.get(n, 0.0) for n in route) > capacity and route:
                # Ejection strategy: eject the node with highest marginal cost
                # to maximize the chance of a better relocation
                worst_idx = 0
                worst_cost = -float("inf")
                for i, node in enumerate(route):
                    prev = route[i - 1] if i > 0 else 0
                    nxt = route[i + 1] if i < len(route) - 1 else 0
                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    if cost > worst_cost:
                        worst_cost = cost
                        worst_idx = i

                pool.append(route.pop(worst_idx))

        depth += 1

    # Final feasibility check: if still infeasible, revert or discard pool
    final_feasible = True
    for route in new_routes:
        if sum(wastes.get(n, 0.0) for n in route) > capacity:
            final_feasible = False
            break

    return [r for r in new_routes if r] if final_feasible else original_routes
