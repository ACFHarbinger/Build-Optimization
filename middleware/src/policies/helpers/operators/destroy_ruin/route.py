"""
Route Removal Operator Module.

Selects an entire route and removes all its customers, placing them into
the unassigned pool.  Route selection is parameterized by strategy:

- ``"random"``: Select a route uniformly at random.
- ``"smallest"``: Select the route with the fewest customers.
- ``"costliest"``: Select the route with the highest cost per customer.
- ``"profitable"``: Select the route with the lowest profit (sum of wastes - cost).

Also includes dedicated profit-based route removal for VRPP problems with
additional strategies:

- ``"worst_profit"``: Remove route with lowest total profit.
- ``"lowest_efficiency"``: Remove route with worst profit-per-node ratio.
- ``"negative_profit"``: Remove unprofitable routes (negative profit).

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.route import route_removal
    >>> routes, removed = route_removal(routes, strategy="smallest")
    >>> from logic.src.policies.helpers.operators.destroy.route import route_profit_removal
    >>> routes, removed = route_profit_removal(routes, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def route_removal(
    routes: List[List[int]],
    strategy: str = "random",
    dist_matrix: Optional[np.ndarray] = None,
    wastes: Optional[Dict[int, float]] = None,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove an entire route from the solution based on cost/simple criteria.

    Args:
        routes: Current solution (list of routes).
        strategy: Selection strategy — ``"random"``, ``"smallest"``,
                  ``"costliest"``, or ``"profitable"``.
        dist_matrix: Distance matrix (required for ``"costliest"`` and ``"profitable"``).
        wastes: Waste look-up (required for ``"profitable"``).
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if not routes:
        return routes, []

    if rng is None:
        rng = Random()

    target = _select_route(routes, strategy, dist_matrix, wastes, rng)
    if target < 0:
        return routes, []

    removed = list(routes[target])
    routes.pop(target)

    # Clean remaining empty routes
    routes = [r for r in routes if r]
    return routes, removed


def _select_route(
    routes: List[List[int]],
    strategy: str,
    dist_matrix: Optional[np.ndarray],
    wastes: Optional[Dict[int, float]],
    rng: Random,
) -> int:
    """Return the index of the route to remove."""
    non_empty = [(i, r) for i, r in enumerate(routes) if r]
    if not non_empty:
        return -1

    if strategy == "random":
        return rng.choice([i for i, _ in non_empty])

    if strategy == "smallest":
        return min(non_empty, key=lambda x: len(x[1]))[0]

    if strategy == "costliest":
        if dist_matrix is None:
            return rng.choice([i for i, _ in non_empty])
        best_idx = -1
        best_cpc = -1.0  # cost-per-customer
        for i, route in non_empty:
            cost = _route_cost(route, dist_matrix)
            cpc = cost / max(len(route), 1)
            if cpc > best_cpc:
                best_cpc = cpc
                best_idx = i
        return best_idx

    if strategy == "profitable":
        if dist_matrix is None or wastes is None:
            return rng.choice([i for i, _ in non_empty])
        worst_idx = -1
        worst_profit = float("inf")
        for i, route in non_empty:
            cost = _route_cost(route, dist_matrix)
            total_waste = sum(wastes.get(n, 0) for n in route)
            profit = total_waste - cost
            if profit < worst_profit:
                worst_profit = profit
                worst_idx = i
        return worst_idx

    return rng.choice([i for i, _ in non_empty])


def _route_cost(route: List[int], dist_matrix: np.ndarray) -> float:
    """Compute total edge cost of depot → route → depot."""
    if not route:
        return 0.0
    cost = dist_matrix[0, route[0]]
    for i in range(len(route) - 1):
        cost += dist_matrix[route[i], route[i + 1]]
    cost += dist_matrix[route[-1], 0]
    return float(cost)


def route_profit_removal(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    strategy: str = "worst_profit",
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove an entire route based on profit-based selection strategies (VRPP).

    Args:
        routes: Current solution (list of routes).
        dist_matrix: Distance matrix including depot at index 0.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        strategy: Selection strategy — ``"worst_profit"``,
                  ``"lowest_efficiency"``, ``"negative_profit"``, or ``"random"``.
        rng: Random number generator.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if not routes:
        return routes, []

    if rng is None:
        rng = Random()

    target = _select_profit_route(routes, strategy, dist_matrix, wastes, R, C, rng)
    if target < 0:
        return routes, []

    removed = list(routes[target])
    routes.pop(target)

    # Clean remaining empty routes
    routes = [r for r in routes if r]
    return routes, removed


def _select_profit_route(
    routes: List[List[int]],
    strategy: str,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    rng: Random,
) -> int:
    """Select route index to remove based on profit-oriented strategy."""
    non_empty = [(i, r) for i, r in enumerate(routes) if r]
    if not non_empty:
        return -1

    if strategy == "random":
        return rng.choice([i for i, _ in non_empty])

    # Calculate metrics for all routes
    metrics = []
    for i, route in non_empty:
        cost = _route_cost(route, dist_matrix) * C
        revenue = sum(wastes.get(n, 0.0) for n in route) * R
        profit = revenue - cost
        efficiency = profit / max(len(route), 1)
        metrics.append({"idx": i, "profit": profit, "efficiency": efficiency})

    if strategy == "worst_profit":
        return int(min(metrics, key=lambda x: x["profit"])["idx"])

    if strategy == "lowest_efficiency":
        return int(min(metrics, key=lambda x: x["efficiency"])["idx"])

    if strategy == "negative_profit":
        unprofitable = [m for m in metrics if m["profit"] < 0]
        if unprofitable:
            return int(min(unprofitable, key=lambda x: x["profit"])["idx"])
        return rng.choice([i for i, _ in non_empty])

    return rng.choice([i for i, _ in non_empty])
