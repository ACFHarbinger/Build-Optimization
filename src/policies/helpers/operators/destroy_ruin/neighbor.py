"""
Neighbor Removal Operator Module.

Selects a random seed node, then removes it along with its ``k-1`` nearest
geographic neighbors from their respective routes.

Also includes profit-based neighbor removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.neighbor import neighbor_removal
    >>> routes, removed = neighbor_removal(routes, n_remove=5, dist_matrix=dm)
    >>> from logic.src.policies.helpers.operators.destroy.neighbor import neighbor_profit_removal
    >>> routes, removed = neighbor_profit_removal(routes, n_remove=5, dist_matrix=dm, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def neighbor_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a seed node and its nearest geographic neighbors.

    Args:
        routes: Current solution (list of routes).
        n_remove: Total number of nodes to remove (including seed).
        dist_matrix: Distance matrix ``(N+1, N+1)`` including depot at index 0.
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random()

    all_nodes = []
    node_loc = {}
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            all_nodes.append(node)
            node_loc[node] = (r_idx, pos)

    if not all_nodes:
        return routes, []

    n_remove = min(n_remove, len(all_nodes))
    seed = rng.choice(all_nodes)

    dists = dist_matrix[seed]
    node_arr = np.array(all_nodes)
    node_dists = dists[node_arr]
    if n_remove >= len(node_arr):
        to_remove = list(node_arr)
    else:
        kth_indices = np.argpartition(node_dists, n_remove)[:n_remove]
        to_remove = list(node_arr[kth_indices])

    removal_plan = []
    for node in to_remove:
        if node in node_loc:
            r_idx, pos = node_loc[node]
            removal_plan.append((r_idx, pos, node))
    removal_plan.sort(key=lambda x: (x[0], x[1]), reverse=True)

    removed = []
    for r_idx, pos, node in removal_plan:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed


def neighbor_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a seed node with low profit and its neighbors with similar profit (VRPP).

    Selects a low-profit seed node (from bottom 25%), then removes nodes with
    the most similar profit values based on the marginal profit formula.

    Args:
        routes: Current solution (list of routes).
        n_remove: Total number of nodes to remove (including seed).
        dist_matrix: Distance matrix ``(N+1, N+1)`` including depot at index 0.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random()

    # 1. Flatten all nodes and pre-compute marginal profits
    all_nodes_data = []
    node_loc = {}
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            all_nodes_data.append((node, profit))
            node_loc[node] = (r_idx, pos)

    if not all_nodes_data:
        return routes, []

    n_remove = min(n_remove, len(all_nodes_data))

    # 2. Sort by profit (lowest first) and pick seed from bottom 25%
    all_nodes_data.sort(key=lambda x: (x[1], x[0]))
    bottom_quartile_size = max(1, len(all_nodes_data) // 4)
    seed_node, seed_profit = all_nodes_data[rng.randint(0, bottom_quartile_size - 1)]

    # 3. Find nodes with most similar profit values
    profit_diffs = []
    for node, profit in all_nodes_data:
        if node == seed_node:
            continue
        profit_diffs.append((node, abs(profit - seed_profit)))

    profit_diffs.sort(key=lambda x: (x[1], x[0]))

    # Take seed + (n_remove-1) most similar profit nodes
    to_remove = [seed_node]
    if n_remove > 1:
        to_remove.extend([x[0] for x in profit_diffs[: n_remove - 1]])

    # 4. Remove
    removal_plan = []
    for node in to_remove:
        if node in node_loc:
            r_idx, pos = node_loc[node]
            removal_plan.append((r_idx, pos, node))
    removal_plan.sort(key=lambda x: (x[0], x[1]), reverse=True)

    removed = []
    for r_idx, pos, node in removal_plan:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
