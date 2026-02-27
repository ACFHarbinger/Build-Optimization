"""
Common initialization heuristic for building geographically compact routes.
"""

import random
from typing import Dict, List

import numpy as np


def build_nn_routes(
    nodes: List[int],
    mandatory_nodes: List[int],
    wastes: Dict[int, float],
    capacity: float,
    dist_matrix: np.ndarray,
    R: float,
    C: float,
) -> List[List[int]]:
    """
    Build geographically compact routes using a Nearest Neighbor logic.

    Instead of purely random packing, this heuristic randomly selects a "seed"
    node for a new route and then sequentially adds the nearest available node
    that fits the capacity. This creates tight geographic clusters.

    Args:
        nodes: Complete list of potential nodes.
        mandatory_nodes: Nodes that MUST be visited.
        wastes: Dictionary of node wastes.
        capacity: Maximum capacity of a route.
        dist_matrix: Distance matrix.
        R: Revenue multiplier.
        C: Cost multiplier.

    Returns:
        List of generated routes.
    """
    mandatory_set = set(mandatory_nodes)

    # Filter profitable nodes + mandatory
    valid_nodes = []
    for node in nodes:
        if node in mandatory_set:
            valid_nodes.append(node)
        else:
            revenue = wastes.get(node, 0.0) * R
            if revenue >= (dist_matrix[0][node] + dist_matrix[node][0]) * C:
                valid_nodes.append(node)

    remaining = set(valid_nodes)
    routes: List[List[int]] = []

    while remaining:
        # Start a new route with a random available node to maintain diversity
        seed = random.choice(list(remaining))
        remaining.remove(seed)

        curr_route = [seed]
        load = wastes.get(seed, 0.0)
        curr_node = seed

        while True:
            # Find the nearest neighbor that fits
            best_n = None
            best_dist = float("inf")

            for n in remaining:
                w = wastes.get(n, 0.0)
                if load + w <= capacity:
                    d = dist_matrix[curr_node][n]
                    if d < best_dist:
                        best_dist = d
                        best_n = n

            if best_n is None:
                break

            curr_route.append(best_n)
            load += wastes.get(best_n, 0.0)
            curr_node = best_n
            remaining.remove(best_n)

        routes.append(curr_route)

    return routes
