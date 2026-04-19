"""
Scramble Mutation Module.

Implements the scramble mutation operator for genetic algorithms applied to VRP
solutions. A random contiguous subsequence within a randomly chosen route is
shuffled in-place, producing higher positional disorder than inversion mutation.

Unlike inversion (which deterministically reverses), scramble applies a random
permutation to the selected segment, making it more suitable for escaping deep
local optima and maintaining genetic diversity in densely populated solution
landscapes.

Algorithm:
1. Select a random route from the solution.
2. Choose two random cut points i ≤ j within that route.
3. Shuffle the subsequence route[i:j+1] uniformly at random.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.evolutionary_mutation.scramble import (
    ...     scramble_mutation,
    ... )
    >>> mutated = scramble_mutation(routes, rng=rng)
"""

from random import Random
from typing import Dict, List, Optional, Union

import numpy as np


def scramble_mutation(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    min_segment: int = 2,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Scramble mutation for CVRP (cost minimisation).

    Selects a random subsequence within a randomly chosen route and permutes
    (shuffles) it in-place. Capacity feasibility is preserved because no
    node is removed or added.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix (unused; API symmetry).
        capacity: Vehicle capacity (unchanged by scramble; API symmetry).
        wastes: Node demands (unchanged by scramble; API symmetry).
        min_segment: Minimum length of the segment to scramble (≥ 2).
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution (new list; original is not modified).
    """
    if rng is None:
        rng = Random()

    eligible = [i for i, r in enumerate(routes) if len(r) >= max(min_segment, 2)]
    if not eligible:
        return routes

    result = [r[:] for r in routes]
    r_idx = rng.choice(eligible)
    route = result[r_idx]

    i, j = sorted(rng.sample(range(len(route)), 2))
    segment = route[i : j + 1]
    rng.shuffle(segment)
    route[i : j + 1] = segment
    return result


def scramble_mutation_profit(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    capacity: float,
    wastes: Union[Dict[int, float], np.ndarray],
    revenue: float,
    cost_unit: float,
    min_segment: int = 2,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """
    Scramble mutation for VRPP (profit maximisation).

    Identical to ``scramble_mutation``; capacity constraints are not affected
    by segment scrambling.  Profit may change only via route length variation.

    Args:
        routes: Current solution as a list of routes.
        distance_matrix: N x N distance matrix (API symmetry).
        capacity: Vehicle capacity (API symmetry).
        wastes: Node demands (API symmetry).
        revenue: Revenue per unit waste (API symmetry).
        cost_unit: Cost per unit distance (API symmetry).
        min_segment: Minimum length of the segment to scramble (≥ 2).
        rng: Random number generator.

    Returns:
        List[List[int]]: Mutated solution.
    """
    return scramble_mutation(routes, distance_matrix, capacity, wastes, min_segment=min_segment, rng=rng)
