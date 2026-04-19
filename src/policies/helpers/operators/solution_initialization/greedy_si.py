"""
Greedy Initialization Module.

Creates an initial constructive solution for the VRPP using greedy insertion,
enforcing strict economic termination to drop unprofitable opportunistic nodes.
"""

import random
from typing import Dict, List, Optional

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def _greedy_profit_insertion(
    routes: List[List[int]],
    unvisited_optional: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes_set: set[int],
    rng: random.Random,
) -> List[List[int]]:
    """Internal greedy profit-driven insertion for initialization."""
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    # Reinsert in random order to increase diversity
    unassigned = list(unvisited_optional)
    rng.shuffle(unassigned)

    for node in unassigned:
        node_waste = wastes.get(node, 0)
        revenue = node_waste * R
        is_mandatory = node in mandatory_nodes_set

        best_profit = -float("inf")
        best_r_idx = -1
        best_pos = -1

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                profit = revenue - (cost * C)

                if profit > best_profit:
                    if not is_mandatory and profit < -1e-4:
                        continue
                    best_profit = profit
                    best_r_idx = r_idx
                    best_pos = pos

        # Check new route option
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - (new_cost * C)

        # Hurdle for starting a new route (Speculative Seed)
        # We allow up to 50% of the return-trip cost to be covered by synergy later.
        seed_hurdle = -0.5 * (new_cost * C)
        if new_profit > best_profit and (is_mandatory or new_profit >= seed_hurdle):
            best_profit = new_profit
            best_r_idx = len(routes)
            best_pos = 0

        # Apply insertion if found
        if best_r_idx != -1:
            if best_r_idx == len(routes):
                routes.append([node])
                loads.append(node_waste)
            else:
                routes[best_r_idx].insert(best_pos, node)
                loads[best_r_idx] += node_waste
        elif is_mandatory:
            routes.append([node])
            loads.append(node_waste)

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)


def build_greedy_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """
    Create an initial solution using a simple greedy heuristic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Waste dictionary mapping node ID to waste volume.
        capacity: Vehicle capacity.
        R: Revenue per unit.
        C: Cost per distance unit.
        mandatory_nodes: List of nodes that MUST be visited.
        rng: Random number generator.

    Returns:
        List[List[int]]: Initial routing solution.
    """
    if rng is None:
        rng = random.Random()

    n_nodes = len(dist_matrix) - 1
    # Nodes are assumed to be 1-indexed (0 is depot)
    nodes = list(range(1, n_nodes + 1))

    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    unvisited_mandatory = sorted(list(mandatory_nodes_set))
    unvisited_optional = sorted(list(set(nodes) - mandatory_nodes_set))

    # STAGE 1: Mandatory Nodes First
    # Pack mandatory nodes into routes as tightly as possible
    routes: List[List[int]] = []

    # We use a simple greedy constructive for mandatory nodes
    while unvisited_mandatory:
        current_route = []
        current_load = 0.0
        last_node = 0

        while True:
            best_node = None
            best_dist = float("inf")

            for node in unvisited_mandatory:
                if current_load + wastes.get(node, 0.0) <= capacity:
                    d = dist_matrix[last_node, node]
                    if d < best_dist:
                        best_dist = d
                        best_node = node

            if best_node is not None:
                current_route.append(best_node)
                current_load += wastes.get(best_node, 0.0)
                unvisited_mandatory.remove(best_node)
                last_node = best_node
            else:
                break

        if current_route:
            routes.append(current_route)
        else:
            # Mandatory node exists but doesn't fit in empty vehicle? Should be handled by data validation,
            # but we'll break to avoid infinite loop.
            break

    # STAGE 2: Profit-Driven Optional Filling
    # Use internal logic to fill the gaps in mandatory routes
    # and potentially start new profitable optional-only routes.
    routes = _greedy_profit_insertion(
        routes=routes,
        unvisited_optional=unvisited_optional,
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        mandatory_nodes_set=mandatory_nodes_set,
        rng=rng,
    )

    return routes
