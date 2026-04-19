"""
Worst Removal Operator Module.

This module implements the worst removal heuristic, which removes the nodes
that cause the highest increase in the objective function (cost/distance).

Follows Ropke & Pisinger (2005) with randomization parameter p >= 1.

Also includes profit-based worst removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.worst import worst_removal
    >>> routes, removed = worst_removal(routes, n_remove=5, dist_matrix=d, p=3.0)
    >>> from logic.src.policies.helpers.operators.destroy.worst import worst_profit_removal
    >>> routes, removed = worst_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0, p=3.0)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def worst_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes that contribute most to the current routing cost with randomization.

    Implements Algorithm 3 from Ropke & Pisinger (2006) randomized worst removal:
    - Compute ``cost(i, s) = f(s) - f_{-i}(s)`` for each node i, where
      ``f_{-i}(s)`` is the objective with node i removed entirely (detour saving).
    - Sort nodes by cost savings in **descending** order (highest savings = worst).
    - For each removal, select node at index: ``floor(y^p * |L|)``
      where ``y ~ U(0, 1)`` is a random number and L is the sorted candidate list.
    - Randomization parameter *p* (must satisfy p ≥ 1):
        - p large (e.g. 100) → ``y^p → 0`` → index → 0 → **always selects the
          worst node** (near-deterministic).
        - p = 1 → ``y^1 = y ~ U(0,1)`` → index is **uniformly random** across
          all candidates.
        - Values between 1 and ~10 provide a smooth interpolation.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        p: Determinism parameter p ≥ 1 (Ropke & Pisinger 2006).  Larger values
           bias selection toward the worst node; p = 1 is fully uniform random.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    removed = []
    for _ in range(n_remove):
        # Recalculate costs for remaining nodes
        costs = []
        for r_idx, route in enumerate(routes):
            if len(route) == 0:
                continue

            for i, node in enumerate(route):
                prev = 0 if i == 0 else route[i - 1]
                nex = 0 if i == len(route) - 1 else route[i + 1]

                # Detour cost (savings from removal)
                savings = dist_matrix[prev][node] + dist_matrix[node][nex] - dist_matrix[prev][nex]
                costs.append((r_idx, i, node, savings))

        if not costs:
            break

        # Sort by savings (highest first)
        costs.sort(key=lambda x: x[3], reverse=True)

        # Randomized selection: index = floor(y^p * |L|)
        # Large p → y^p → 0 → index 0 → always picks the worst node.
        # p = 1 → y^1 = y ~ U(0,1) → uniform random index.
        L = len(costs)
        y = rng.random()  # U(0,1)
        idx = min(int(y**p * L), L - 1)

        r_idx, n_idx, node, _ = costs[idx]

        # Remove the selected node
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx] = [n for n in routes[r_idx] if n != node]
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed


def worst_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    p: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with the worst marginal profit contribution with randomization (VRPP).

    Applies Algorithm 3 from Ropke & Pisinger (2006) to a profit-maximisation
    objective by replacing the cost savings criterion with marginal profit:

        profit_i = waste_i * R - detour_cost_i * C

    Nodes are sorted **ascending** by profit (lowest profit = worst contribution)
    and the same randomized index formula ``floor(y^p * |L|)`` is applied.

    Randomization parameter *p* semantics (identical to ``worst_removal``):
        - p large → always selects the node with lowest profit (deterministic).
        - p = 1 → uniformly random selection across all candidates.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        p: Determinism parameter p ≥ 1.  Larger values bias selection toward the
           worst-profit node; p = 1 is fully uniform random.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if rng is None:
        rng = np.random.default_rng()

    removed = []
    for _ in range(n_remove):
        # Recalculate profits for remaining nodes
        profits = []
        for r_idx, route in enumerate(routes):
            if len(route) == 0:
                continue

            for i, node in enumerate(route):
                revenue = wastes.get(node, 0.0) * R
                prev = 0 if i == 0 else route[i - 1]
                nex = 0 if i == len(route) - 1 else route[i + 1]

                detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
                profit = revenue - (detour_cost * C)
                profits.append((r_idx, i, node, profit))

        if not profits:
            break

        # Sort by profit (lowest first = worst nodes)
        profits.sort(key=lambda x: x[3])

        # Randomized selection
        L = len(profits)
        y = rng.random()
        idx = min(int(y**p * L), L - 1)

        r_idx, n_idx, node, _ = profits[idx]

        # Remove the selected node
        if n_idx < len(routes[r_idx]) and routes[r_idx][n_idx] == node:
            routes[r_idx] = [n for n in routes[r_idx] if n != node]
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
