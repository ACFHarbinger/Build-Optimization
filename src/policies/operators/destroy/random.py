"""
Random Removal Operator Module.

This module implements the random removal heuristic, which simply removes
a specified number of nodes chosen uniformly at random.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.destroy.random import random_removal
    >>> routes, removed = random_removal(routes, n_remove=5)
"""

import random
from typing import List, Tuple


def random_removal(routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes randomly from the solution.

    Selects `n_remove` nodes uniformly at random from the current routes
    and removes them.

    Args:
        routes: The current solution (list of routes).
        n_remove: Number of nodes to remove.

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple containing the
        modified routes (with nodes removed) and a list of removed node IDs.
    """
    removed = []
    # Flatten
    all_nodes = []
    for r_idx, r in enumerate(routes):
        for n_idx, node in enumerate(r):
            all_nodes.append((r_idx, n_idx, node))

    if n_remove >= len(all_nodes):
        return [[]], [n for _, _, n in all_nodes]

    targets = random.sample(all_nodes, n_remove)

    # Sort targets by r_idx, n_idx desc to pop safely
    targets.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for r_idx, n_idx, node in targets:
        routes[r_idx].pop(n_idx)
        removed.append(node)

    # Clean empty routes
    routes = [r for r in routes if r]
    return routes, removed
