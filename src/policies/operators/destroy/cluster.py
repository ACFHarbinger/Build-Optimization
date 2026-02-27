"""
Cluster Removal Operator Module.

This module implements the cluster removal heuristic, which removes a cluster of
nodes based on spatial proximity, effectively acting as a variant of Shaw removal.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.destroy.cluster import cluster_removal
    >>> routes, removed = cluster_removal(routes, n_remove=5, dist_matrix=d, nodes=all_nodes)
"""

import random
from typing import List, Tuple

import numpy as np

from .random import random_removal


def cluster_removal(
    routes: List[List[int]], n_remove: int, dist_matrix: np.ndarray, nodes: List[int]
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a cluster of nodes based on spatial proximity (Shaw Removal variant).

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        nodes (List[int]): List of all node IDs.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    # Pick a random node, remove it and its k nearest neighbors
    if not any(routes):
        return routes, []

    # Pick seed
    seed_route_idx = random.randint(0, len(routes) - 1)
    if not routes[seed_route_idx]:
        return random_removal(routes, n_remove)

    seed_node = random.choice(routes[seed_route_idx])

    removed = [seed_node]

    # Get all nodes current pos
    node_map = {}
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            node_map[node] = (r_idx, n_idx)

    # Find neighbors
    candidates = []
    for v in nodes:
        if v == seed_node or v not in node_map:
            continue
        dist = dist_matrix[seed_node][v]
        candidates.append((v, dist))

    candidates.sort(key=lambda x: x[1])

    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed.extend(target_nodes)

    # Now remove them from routes
    to_remove_locs = []
    for node in removed:
        if node in node_map:
            to_remove_locs.append((*node_map[node], node))

    to_remove_locs.sort(key=lambda x: (x[0], x[1]), reverse=True)

    final_removed = []
    for r_idx, n_idx, node in to_remove_locs:
        routes[r_idx].pop(n_idx)
        final_removed.append(node)

    routes = [r for r in routes if r]
    return routes, final_removed
