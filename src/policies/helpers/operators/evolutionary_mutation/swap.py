"""
Swap Mutation Module.

Implements the swap mutation operator for genetic algorithms applied to VRP
solutions. Two randomly selected node positions in the flat chromosome are
exchanged, analogous to a bit-flip for discrete permutation spaces.

This is the most elementary GA mutation primitive and serves as the stochastic
baseline against which more structured mutations are compared.

Algorithm:
1. Flatten the solution into a chromosome (nodes with -1 depot separators).
2. Select two random non-separator positions.
3. Swap the gene values at those positions.
4. Decode the mutated chromosome back into a route plan.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.evolutionary_mutation.swap import (
    ...     swap_mutation,
    ... )
    >>> mutated = swap_mutation(routes, distance_matrix, capacity, wastes, rng=rng)
"""

from random import Random
from typing import Dict, List, Optional, Union

import numpy as np


def swap_mutation(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    n_swaps: int = 1,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Swap mutation for CVRP (cost minimisation).

    Randomly exchanges the positions of two distinct non-depot nodes
    within the flat chromosome representation of the solution.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix.
        capacity: Vehicle capacity constraint.
        wastes: Node demands (dict or 1-D array).
        n_swaps: Number of independent swap operations to apply.
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution.
    """
    if rng is None:
        rng = Random()

    if not routes:
        return routes

    chromosome, n_vehicles = _encode(routes)
    nodes_idx = [i for i, g in enumerate(chromosome) if g != -1]

    if len(nodes_idx) < 2:
        return routes

    for _ in range(n_swaps):
        i, j = rng.sample(nodes_idx, 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    return _decode(chromosome, n_vehicles)


def swap_mutation_profit(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    revenue: float,
    cost_unit: float,
    n_swaps: int = 1,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Swap mutation for VRPP (profit maximisation).

    Same gene-swap logic as ``swap_mutation`` with an additional post-mutation
    capacity repair: if a vehicle exceeds capacity after the swap, the least
    profitable node in the overloaded route is ejected.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix.
        capacity: Vehicle capacity constraint.
        wastes: Node demands (dict or 1-D array).
        revenue: Revenue per unit of waste collected.
        cost_unit: Travel cost per unit distance.
        n_swaps: Number of independent swap operations to apply.
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution.
    """
    if rng is None:
        rng = Random()

    if not routes:
        return routes

    chromosome, n_vehicles = _encode(routes)
    nodes_idx = [i for i, g in enumerate(chromosome) if g != -1]

    if len(nodes_idx) < 2:
        return routes

    for _ in range(n_swaps):
        i, j = rng.sample(nodes_idx, 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

    decoded = _decode(chromosome, n_vehicles)
    return _repair_capacity_profit(decoded, capacity, wastes, revenue, cost_unit, distance_matrix)


# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------


def _encode(routes: List[List[int]]) -> tuple:
    """Encode routes as flat chromosome with -1 depot separators."""
    chromosome: List[int] = []
    for i, route in enumerate(routes):
        chromosome.extend(route)
        if i < len(routes) - 1:
            chromosome.append(-1)
    return chromosome, len(routes)


def _decode(chromosome: List[int], n_vehicles: int) -> List[List[int]]:
    """Decode a flat chromosome back to routes, preserving vehicle count."""
    routes: List[List[int]] = []
    current: List[int] = []
    for gene in chromosome:
        if gene == -1:
            routes.append(current)
            current = []
        else:
            current.append(gene)
    routes.append(current)
    return [r for r in routes if r]


def _get_demand(wastes: Union[Dict[int, float], np.ndarray], node: int) -> float:
    if isinstance(wastes, dict):
        return wastes.get(node, 0.0)
    return float(wastes[node]) if node < len(wastes) else 0.0


def _repair_capacity_profit(
    routes: List[List[int]],
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    revenue: float,
    cost_unit: float,
    dist: np.ndarray,
) -> List[List[int]]:
    """
    Repair capacity violations post-mutation by ejecting the least profitable
    node from any overloaded route.
    """
    repaired: List[List[int]] = []
    for route in routes:
        load = sum(_get_demand(wastes, n) for n in route)
        while load > capacity + 1e-6 and route:
            # Eject the node with the worst marginal profit
            worst_idx = 0
            worst_profit = float("inf")
            for pos, node in enumerate(route):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos + 1] if pos < len(route) - 1 else 0
                cost_saving = dist[prev, node] + dist[node, nxt] - dist[prev, nxt]
                marginal = _get_demand(wastes, node) * revenue - cost_saving * cost_unit
                if marginal < worst_profit:
                    worst_profit = marginal
                    worst_idx = pos
            load -= _get_demand(wastes, route[worst_idx])
            route.pop(worst_idx)
        if route:
            repaired.append(route)
    return repaired
