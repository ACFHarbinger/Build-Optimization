"""
Greedy Blink Insertion Operator Module.

This module implements the 'Greedy with Blinks' insertion heuristic. It is a
randomized version of greedy insertion where the algorithm 'blinks' (skips)
the best position with some probability, encouraging diversity.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.repair.greedy_blink import greedy_insertion_with_blinks
    >>> routes = greedy_insertion_with_blinks(routes, removed, dist_matrix, wastes, capacity, blink_rate=0.1)
"""

import random
from typing import Dict, List

import numpy as np


def greedy_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    blink_rate: float = 0.1,
) -> List[List[int]]:
    """
    Greedy insertion with randomized skips ('blinks').

    Similar to standard greedy insertion, but during the selection of the best
    position, the algorithm may skip the absolute best option with probability
    `blink_rate` and choose the second best, and so on.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        wastes: wastes.
        capacity: Capacity.
        blink_rate: Probability of skipping a check.

    Returns:
        Routes with all nodes reinserted.
    """
    loads = []
    for route in routes:
        loads.append(sum(wastes.get(n, 0) for n in route))

    # Reinsert in random order
    random.shuffle(removed_nodes)

    for node in removed_nodes:
        waste = wastes.get(node, 0)
        best_cost = float("inf")
        best_r_idx = -1
        best_pos = -1

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + waste > capacity:
                continue

            for pos in range(len(route) + 1):
                # Blink check
                if random.random() < blink_rate:
                    continue

                prev = 0 if pos == 0 else route[pos - 1]
                nex = 0 if pos == len(route) else route[pos]

                cost = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]

                if cost < best_cost:
                    best_cost = cost
                    best_r_idx = r_idx
                    best_pos = pos

        # Check new route
        new_route_cost = dist_matrix[0][node] + dist_matrix[node][0]
        if new_route_cost < best_cost:
            best_cost = new_route_cost
            best_r_idx = len(routes)
            best_pos = 0

        # Apply
        if best_r_idx == len(routes):
            routes.append([node])
            loads.append(waste)
        else:
            routes[best_r_idx].insert(best_pos, node)
            loads[best_r_idx] += waste

    return routes
