"""
Individual Representation for the HGS-ADC MPVRP solver.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Individual:
    """
    Representation of an individual in the HGS-ADC population.

    The search space for patterns is fully open (unconstrained).
    Any node `i` can have any $p_i$ from `0` to `2^T - 1`.

    Fields:
        patterns: np.ndarray of length N. p_i is an integer describing visit days via bits.
                  (e.g., bit 0 corresponds to day 0).
        giant_tours: List of T np.ndarray permutations containing only active nodes for day t.
                     Does not include the depot (node 0) in the pattern or tour permutations inherently,
                     or rather, node 0 is handled implicitly.
        cost: Sum of routing costs for all T days.
        capacity_violations: Delta Q sum for all T days.
        fit: Raw fitness.
        dc: Diversity contribution.
        biased_fitness: Biased fitness BF.
        is_feasible: True if capacity_violations == 0.
        routes: The decoded fleet routes for T days. [Day][Vehicle][Nodes].
    """

    patterns: np.ndarray
    giant_tours: List[np.ndarray]

    cost: float = 0.0
    capacity_violations: float = 0.0
    fit: float = 0.0
    dc: float = 0.0
    biased_fitness: float = 0.0
    is_feasible: bool = False

    routes: List[List[List[int]]] = field(default_factory=list)

    def is_active(self, node: int, day_t: int) -> bool:
        """Check if node is active on a specific day [0, T-1] based on its bit."""
        return bool((self.patterns[node] >> day_t) & 1)

    def set_active(self, node: int, day_t: int, active: bool) -> None:
        """Set the active status of a node on a specific day."""
        if active:
            self.patterns[node] |= 1 << day_t
        else:
            self.patterns[node] &= ~(1 << day_t)
