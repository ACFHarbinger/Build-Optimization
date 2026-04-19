"""
GRASP construction heuristic for VRPP initialization.

Reference:
    Feo, T. A., & Resende, M. G. C. (1995). Greedy randomized adaptive search
    procedures. Journal of Global Optimization, 6(2), 109-133.
"""

import random
from typing import Dict, List, Optional

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def build_grasp_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    alpha: float = 0.15,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """
    Build routes using GRASP construction.

    At each insertion step, compute the marginal profit for inserting every
    unrouted node at its best feasible position. Build the Restricted
    Candidate List (RCL): all nodes with marginal profit ≥
        profit_min + (1 - alpha) * (profit_max - profit_min).
    Select uniformly from the RCL.

    Args:
        dist_matrix:     (N+1)×(N+1) distance matrix.
        wastes:          {node_id → fill level}.
        capacity:        Vehicle capacity Q.
        R:               Revenue per unit waste.
        C:               Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        alpha:           RCL greediness: 0.0 = fully greedy, 1.0 = fully random.
        rng:             Random generator.

    Returns:
        List of routes (depot excluded).
    """
    if rng is None:
        rng = random.Random()

    mandatory_set = set(mandatory_nodes or [])
    n = len(dist_matrix) - 1
    all_nodes = list(range(1, n + 1))

    eligible = [
        i
        for i in all_nodes
        if i in mandatory_set or wastes.get(i, 0.0) * R >= (dist_matrix[0, i] + dist_matrix[i, 0]) * C
    ]

    routes: List[List[int]] = []
    loads: List[float] = []
    unrouted = list(eligible)

    while unrouted:
        # Compute best marginal profit per node
        node_scores = []
        for node in unrouted:
            node_waste = wastes.get(node, 0.0)
            node_rev = node_waste * R
            best_delta = -float("inf")

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = node_rev - (dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]) * C
                    if delta > best_delta:
                        best_delta = delta

            # New route option
            solo = node_rev - (dist_matrix[0, node] + dist_matrix[node, 0]) * C
            if solo > best_delta:
                best_delta = solo

            node_scores.append((node, best_delta))

        # Build RCL
        profits = [s for _, s in node_scores]
        p_min, p_max = min(profits), max(profits)
        threshold = p_min + (1.0 - alpha) * (p_max - p_min)
        rcl = [node for node, s in node_scores if s >= threshold]

        chosen = rng.choice(rcl)
        unrouted.remove(chosen)
        node_waste = wastes.get(chosen, 0.0)
        node_rev = node_waste * R

        # Insert at best feasible position
        best_cost = float("inf")
        best_r = len(routes)
        best_pos = 0

        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost = (dist_matrix[prev, chosen] + dist_matrix[chosen, nxt] - dist_matrix[prev, nxt]) * C - node_rev
                if cost < best_cost:
                    best_cost = cost
                    best_r, best_pos = r_idx, pos

        solo_cost = (dist_matrix[0, chosen] + dist_matrix[chosen, 0]) * C - node_rev
        if solo_cost < best_cost:
            best_r, best_pos = len(routes), 0

        if best_r == len(routes):
            routes.append([chosen])
            loads.append(node_waste)
        else:
            routes[best_r].insert(best_pos, chosen)
            loads[best_r] += node_waste

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
