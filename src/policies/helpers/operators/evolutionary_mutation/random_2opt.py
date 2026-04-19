"""
Random 2-opt Mutation Module.

Implements an *unguided* 2-opt mutation for genetic algorithms.  Unlike the
steepest-descent 2-opt in ``improvement_descent``, this operator applies the
2-opt reversal at randomly chosen cut points without checking whether the move
improves tour cost.

The deliberate disregard for improvement makes this a true stochastic mutation:
it explores the neighbourhood at random, enabling the GA to escape local optima
that gradient-following operators cannot leave.

Algorithm:
1. Select a route at random (or the route containing the most nodes).
2. Choose two random cut points i < j within that route.
3. Reverse the segment route[i:j+1] (standard 2-opt reconnection).

References:
    Lin, S. (1965). Computer solutions of the travelling salesman problem.
    Bell System Technical Journal, 44(10), 2245–2269.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.evolutionary_mutation.random_2opt import (
    ...     random_2opt_mutation,
    ... )
    >>> mutated = random_2opt_mutation(routes, distance_matrix, capacity, wastes)
"""

from random import Random
from typing import Dict, List, Optional, Union

import numpy as np


def random_2opt_mutation(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    n_moves: int = 1,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Unguided random 2-opt mutation for CVRP (cost minimisation).

    Reverses a random segment of a randomly chosen route without any
    improvement check.  This is a stochastic diversification move; it may
    worsen the objective.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix (unused; API symmetry).
        capacity: Vehicle capacity (unchanged by 2-opt; API symmetry).
        wastes: Node demands (unchanged by 2-opt; API symmetry).
        n_moves: Number of random 2-opt moves to apply.
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

    for _ in range(n_moves):
        r_idx = rng.choice(eligible)
        route = result[r_idx]
        if len(route) < 2:
            continue
        i, j = sorted(rng.sample(range(len(route)), 2))
        route[i : j + 1] = route[i : j + 1][::-1]

    return result


def random_2opt_mutation_profit(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    revenue: float,
    cost_unit: float,
    n_moves: int = 1,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Unguided random 2-opt mutation for VRPP (profit maximisation).

    Identical to ``random_2opt_mutation``; 2-opt segment reversal never
    alters which nodes are visited, so capacity constraints are unaffected.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix (API symmetry).
        capacity: Vehicle capacity (API symmetry).
        wastes: Node demands (API symmetry).
        revenue: Revenue per unit waste (API symmetry).
        cost_unit: Cost per unit distance (API symmetry).
        n_moves: Number of random 2-opt moves to apply.
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution.
    """
    return random_2opt_mutation(routes, distance_matrix, capacity, wastes, n_moves=n_moves, rng=rng)
