"""
Random Removal Operator Module.

This module implements the random removal heuristic, which simply removes
a specified number of nodes chosen uniformly at random.

Also includes profit-based random removal (biased random) for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.random import random_removal
    >>> routes, removed = random_removal(routes, n_remove=5)
    >>> from logic.src.policies.helpers.operators.destroy.random import random_profit_removal
    >>> routes, removed = random_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def random_removal(
    routes: List[List[int]], n_remove: int, rng: Optional[Random] = None
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes randomly from the solution.

    Selects `n_remove` nodes uniformly at random from the current routes
    and removes them.

    Args:
        routes: The current solution (list of routes).
        n_remove: Number of nodes to remove.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple containing the
        modified routes (with nodes removed) and a list of removed node IDs.
    """
    # Flatten
    all_nodes = [n for r in routes for n in r]

    if not all_nodes:
        return routes, []

    if n_remove >= len(all_nodes):
        return [[] for _ in routes], all_nodes

    if rng is None:
        rng = Random()

    removed = rng.sample(all_nodes, n_remove)
    removed_set: Set[int] = set(removed)

    # Efficient route modification: filter out removed nodes in a single pass
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


def random_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    bias_strength: float = 3.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    r"""
    Remove nodes with biased-random selection favoring low-profit nodes (VRPP).

    Uses marginal profit-weighted sampling with NumPy vectorization ($O(N \log N)$).
    Nodes with lower profit contribution have higher probability of removal.

    Args:
        routes: The current solution (list of routes).
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        bias_strength: Controls bias toward low-profit nodes (default 3.0).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed node IDs.
    """
    # 1. Flatten and calculate marginal profits
    all_nodes = []
    node_profits = []
    for route in routes:
        for n_idx, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if n_idx == 0 else route[n_idx - 1]
            nex = 0 if n_idx == len(route) - 1 else route[n_idx + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            all_nodes.append(node)
            node_profits.append(profit)

    if not all_nodes:
        return routes, []

    if n_remove >= len(all_nodes):
        return [[] for _ in routes], all_nodes

    # 2. Vectorized profit-biased sampling
    node_profits_arr = np.array(node_profits)
    min_p, max_p = node_profits_arr.min(), node_profits_arr.max()
    p_range = max_p - min_p if max_p != min_p else 1.0

    # Weights: lower profit -> higher weight
    norm_low_profit = (max_p - node_profits_arr) / p_range
    weights = (norm_low_profit + 0.1) ** bias_strength
    total_w = weights.sum()

    # np.random.choice defaults to uniform if p is None
    probs = None if total_w == 0 else weights / total_w

    # Use a high-quality entropy seed derived from the input rng to maintain reproducibility
    # np.random.default_rng expects an int or sequence of ints, not a bound method
    entropy_seed = rng.randint(0, 2**31 - 1) if rng else 42
    np_rng = np.random.default_rng(entropy_seed)

    selected_indices = np_rng.choice(len(all_nodes), size=n_remove, replace=False, p=probs)
    removed = [all_nodes[i] for i in selected_indices]
    removed_set: Set[int] = set(removed)

    # 3. Efficient route modification
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
