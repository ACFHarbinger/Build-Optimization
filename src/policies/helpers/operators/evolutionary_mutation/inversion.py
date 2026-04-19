"""
Inversion Mutation Module.

Implements the inversion mutation operator (also known as segment reversal) for
genetic algorithms applied to VRP solutions. A random contiguous subsequence
within a single route is reversed in-place, disrupting local ordering without
altering the set of visited nodes.

This is the permutation-space analogue of bit-string inversion and is a
foundation of many evolutionary strategies including SA-style perturbation and
Differential Evolution step approximations.

Algorithm:
1. Select a random route from the solution.
2. Choose two random cut points i < j within that route.
3. Reverse the subsequence route[i:j+1].

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.evolutionary_mutation.inversion import (
    ...     inversion_mutation,
    ... )
    >>> mutated = inversion_mutation(routes, rng=rng)
"""

from random import Random
from typing import Dict, List, Optional, Union

import numpy as np


def inversion_mutation(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Inversion mutation for CVRP (cost minimisation).

    Reverses a random contiguous subsequence within a randomly chosen route.
    The mutation is always feasible because no node is added or removed and
    route loads are unchanged.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix (unused; kept for API symmetry).
        capacity: Vehicle capacity (unused; kept for API symmetry).
        wastes: Node demands (unused; kept for API symmetry).
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution (new list; original is not modified).
    """
    if rng is None:
        rng = Random()

    eligible = [i for i, r in enumerate(routes) if len(r) >= 2]
    if not eligible:
        return routes

    result = [r[:] for r in routes]
    r_idx = rng.choice(eligible)
    route = result[r_idx]

    i, j = sorted(rng.sample(range(len(route)), 2))
    route[i : j + 1] = route[i : j + 1][::-1]
    return result


def inversion_mutation_profit(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    revenue: float,
    cost_unit: float,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Inversion mutation for VRPP (profit maximisation).

    Same segment-reversal logic as ``inversion_mutation``.  Because inversion
    neither adds nor removes nodes, it is always capacity-feasible.  The profit
    objective changes only through route cost; no repair is required.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix (unused directly; symmetry).
        capacity: Vehicle capacity (unchanged by inversion; symmetry).
        wastes: Node demands (unchanged by inversion; symmetry).
        revenue: Revenue per unit waste (unused directly; symmetry).
        cost_unit: Cost per unit distance (unused directly; symmetry).
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution.
    """
    return inversion_mutation(routes, distance_matrix, capacity, wastes, rng=rng)


def _get_demand(wastes: Union[Dict[int, float], np.ndarray], node: int) -> float:
    if isinstance(wastes, dict):
        return wastes.get(node, 0.0)
    return float(wastes[node]) if node < len(wastes) else 0.0
