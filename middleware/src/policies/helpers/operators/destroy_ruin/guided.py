"""
Guided Removal Operator Module for G-LNS.

This module implements the penalized removal heuristic, which targets nodes
involved in highly penalized edges as defined by Guided Large Neighborhood
Search (G-LNS). This helps the search explicitly break away from local
optima by removing features that the G-LNS mechanism has identified as
undesirable. This operator serves as the G-LNS equivalent of Fast Local
Search (FLS).

Reference:
    Voudouris, C., & Tsang, E. "Guided Local Search and Its
    Application to the Traveling Salesman Problem", 1999.
"""

from random import Random
from typing import List, Optional, Tuple

import numpy as np


def penalized_removal(
    routes: List[List[int]],
    n_remove: int,
    penalties: np.ndarray,
    p: float = 1.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes associated with highly penalized edges in G-LNS.

    This operator serves as the stochastic LNS equivalent of Fast Local Search
    (FLS) sub-neighborhood activation. By scoring each node based on the sum
    of penalties of its incident edges and forcing the LNS ruin operator to
    deterministically bias the removal of these high-scoring nodes, it achieves
    the exact theoretical goal of FLS: focusing computational effort strictly
    on tearing down penalized features while allowing regret-based insertion
    to repair the sub-neighborhood.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        penalties: Edge penalty matrix from GLSSolver.
        p: Randomness parameter (p >= 1). Higher p = more deterministic.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if rng is None:
        rng = Random()

    removed = []
    for _ in range(n_remove):
        # Recalculating the penalty exposures sequentially is a deliberate
        # theoretical choice. Because removing a node connects its predecessor
        # and successor, the active penalties in the route cascade. Dynamic
        # recalculation ensures topological consistency during the ruin phase,
        # and the $O(k \cdot N)$ overhead is negligible since the number of
        # removals $k$ is small.
        node_scores = []
        for r_idx, route in enumerate(routes):
            if not route:
                continue

            for i, node in enumerate(route):
                prev = 0 if i == 0 else route[i - 1]
                nex = 0 if i == len(route) - 1 else route[i + 1]

                # Sum of penalties on adjacent edges
                score = penalties[prev, node] + penalties[node, nex]
                node_scores.append((r_idx, i, node, score))

        if not node_scores:
            break

        # Sort by penalty score (highest first)
        node_scores.sort(key=lambda x: x[3], reverse=True)

        # Randomized selection: index = floor(y^p * |L|)
        L = len(node_scores)
        y = rng.random()
        idx = min(int(y**p * L), L - 1)

        r_idx, n_idx, node, _ = node_scores[idx]

        # Remove the selected node
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx] = [n for n in routes[r_idx] if n != node]
            removed.append(node)

    # Clean up empty routes
    routes = [r for r in routes if r]
    return routes, removed
