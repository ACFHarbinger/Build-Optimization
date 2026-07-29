"""
Profit-Similarity and Cluster Removal Operator Module.

This module implements removal heuristics based on spatial and attribute similarity.
Includes standard cluster removal (spatial) and profit-similarity removal (VRPP).

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.cluster import cluster_removal
    >>> routes, removed = cluster_removal(routes, n_remove=5, dist_matrix=d, nodes=all_nodes)
    >>> from logic.src.policies.helpers.operators.destroy.cluster import cluster_profit_removal
    >>> routes, removed = cluster_profit_removal(routes, n_remove=5, dist_matrix=d,
    ...                                              wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def cluster_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    nodes: List[int],
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a geographic cluster of nodes based on spatial proximity.

    **Paper Reference**: Pisinger & Ropke (2007), §5.1.4 — *Cluster Removal*.

    **Relationship to the Paper**: The cluster removal heuristic in the paper
    runs Kruskal's MST algorithm on the relatedness graph (using edge weight
    ``r_{ij}``) and terminates when two connected components remain.  One
    component is chosen at random and its nodes are removed.  If fewer than
    ``q`` nodes have been selected, the process is repeated on a different
    route.

    **This implementation** is a geographic-proximity simplification: it picks
    a random seed node and removes the ``n_remove - 1`` spatially closest
    nodes.  This is equivalent to the single-cluster phase of the paper but
    omits the iterative multi-route extension and the explicit MST computation.
    The simplification is appropriate for VRPP instances with a homogeneous
    fleet, where inter-request distance is the dominant relatedness component.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        nodes (List[int]): List of all node IDs.
        rng (Optional[Random]): Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes):
        return routes, []

    if rng is None:
        rng = Random()

    # Pick seed
    all_visited = [n for r in routes for n in r]
    if not all_visited:
        return routes, []
    seed_node = rng.choice(all_visited)

    removed = [seed_node]

    # Find geographic neighbors
    candidates = []
    for v in nodes:
        if v == seed_node or v not in all_visited:
            continue
        dist = dist_matrix[seed_node][v]
        candidates.append((v, dist))

    candidates.sort(key=lambda x: (x[1], x[0]))
    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed.extend(target_nodes)

    # Efficient route modification: filter out removed nodes in a single pass
    removed_set: Set[int] = set(removed)
    final_removed = []

    modified_routes = []
    for r in routes:
        new_route = []
        for node in r:
            if node in removed_set:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed


def cluster_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a group of nodes based on similarity in their profit contributions.

    Unlike standard Cluster Removal which uses spatial proximity (Pisinger & Ropke 2007),
    this is a novel VRPP adaptation that destroys routes by targeting nodes with
    similar marginal profit levels, effectively isolating regions of equivalent
    unprofitability for the search to re-evaluate.

    Args:
        routes (List[List[int]]): Current routes.
        n_remove (int): Number of nodes to remove.
        dist_matrix (np.ndarray): Distance matrix.
        wastes (Dict[int, float]): Waste/profit values for each node.
        R (float): Revenue per unit waste.
        C (float): Cost per unit distance.
        rng (Optional[Random]): Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes):
        return routes, []

    if rng is None:
        rng = Random()

    # 1. Pre-calculate marginal profits
    all_nodes_data = []
    for _, route in enumerate(routes):
        for pos, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            all_nodes_data.append((node, profit))

    if not all_nodes_data:
        return routes, []

    # 2. Sort by profit and pick seed from bottom 25%
    all_nodes_data.sort(key=lambda x: (x[1], x[0]))
    bottom_quartile_size = max(1, len(all_nodes_data) // 4)
    seed_node, seed_profit = all_nodes_data[rng.randint(0, bottom_quartile_size - 1)]

    # 3. Find nodes with similar profit
    candidates = []
    for node, profit in all_nodes_data:
        if node == seed_node:
            continue
        candidates.append((node, abs(profit - seed_profit)))

    candidates.sort(key=lambda x: (x[1], x[0]))
    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed = [seed_node] + target_nodes

    # 4. Efficient route modification
    removed_set: Set[int] = set(removed)
    final_removed = []
    modified_routes = []
    for r in routes:
        new_route = []
        for node in r:
            if node in removed_set:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed
