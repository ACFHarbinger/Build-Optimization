"""
Regret-based insertion heuristic for VRPP initialization.

Reference:
    Potvin, J.-Y., & Rousseau, J.-M. (1993). A parallel route building algorithm
    for the vehicle routing and time window problem.
    European Journal of Operational Research, 66(3), 331-340.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def _best_insertion_cost(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
) -> Tuple[float, int, int]:
    """
    Return (best_delta_cost, route_idx, position) for inserting `node`.

    delta_cost = insertion distance increase minus marginal revenue.
    Lower is better (more profitable insertion).
    Returns (inf, -1, -1) if no feasible insertion exists.
    """
    node_waste = wastes.get(node, 0.0)
    node_rev = node_waste * R
    best_cost = math.inf
    best_r = -1
    best_pos = -1

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + node_waste > capacity:
            continue
        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            delta_dist = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
            delta_cost = delta_dist * C - node_rev
            if delta_cost < best_cost:
                best_cost = delta_cost
                best_r = r_idx
                best_pos = pos

    # Also consider opening a new route
    new_dist = dist_matrix[0, node] + dist_matrix[node, 0]
    new_cost = new_dist * C - node_rev
    if new_cost < best_cost:
        best_cost = new_cost
        best_r = len(routes)  # sentinel: open new route
        best_pos = 0

    return best_cost, best_r, best_pos


def build_regret_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    regret_k: int = 2,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """
    Build routes using regret-k insertion.

    At each step, for every unrouted node compute the cost difference between
    its k-th best and best feasible insertion positions (its "regret").
    Insert the node with the highest regret first.

    Args:
        dist_matrix:     (N+1)×(N+1) distance matrix.
        wastes:          {node_id → fill level}.
        capacity:        Vehicle capacity Q.
        R:               Revenue per unit waste.
        C:               Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        regret_k:        Number of alternatives to compare (k=2 is standard).
        rng:             Random generator (used for tie-breaking).

    Returns:
        List of routes (depot excluded).
    """
    if rng is None:
        rng = random.Random()

    mandatory_set = set(mandatory_nodes or [])
    n = len(dist_matrix) - 1
    all_nodes = list(range(1, n + 1))

    # Filter by profitability (mandatory nodes always included)
    eligible = [
        i
        for i in all_nodes
        if i in mandatory_set or wastes.get(i, 0.0) * R >= (dist_matrix[0, i] + dist_matrix[i, 0]) * C
    ]

    routes: List[List[int]] = []
    loads: List[float] = []
    unrouted = list(eligible)

    while unrouted:
        # Compute regret for each unrouted node
        best_to_insert = None
        best_regret = -math.inf

        for node in unrouted:
            # Collect the top-k insertion costs
            all_costs = []
            node_waste = wastes.get(node, 0.0)
            node_rev = node_waste * R

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = (dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]) * C - node_rev
                    all_costs.append((delta, r_idx, pos))

            # New route option
            new_cost = (dist_matrix[0, node] + dist_matrix[node, 0]) * C - node_rev
            all_costs.append((new_cost, len(routes), 0))

            all_costs.sort(key=lambda t: t[0])

            if not all_costs:
                continue

            regret = (
                all_costs[regret_k - 1][0] - all_costs[0][0]
                if len(all_costs) >= regret_k
                else -all_costs[0][0]  # penalise scarcity of options
            )

            if regret > best_regret or (regret == best_regret and rng.random() < 0.5):
                best_regret = regret
                best_to_insert = (node, all_costs[0][1], all_costs[0][2])

        if best_to_insert is None:
            break

        node, r_idx, pos = best_to_insert
        unrouted.remove(node)
        node_waste = wastes.get(node, 0.0)

        if r_idx == len(routes):
            routes.append([node])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, node)
            loads[r_idx] += node_waste

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
