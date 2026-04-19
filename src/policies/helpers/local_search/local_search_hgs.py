"""
HGS Local Search Module.

This module provides the local search implementation specifically for
Hybrid Genetic Search (HGS). It applies operators to HGS 'Individual' objects.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
    >>> ls = HGSLocalSearch(dist_matrix, waste, capacity, R, C, params)
    >>> improved_ind = ls.optimize(individual)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

from .local_search_base import LocalSearch

if TYPE_CHECKING:
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams


class HGSLocalSearch(LocalSearch):
    """
    Local Search module for HGS.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: "HGSParams",
    ):
        """
        Initialize HGS Local Search.

        Args:
            dist_matrix: Distance matrix.
            waste: Dictionary of waste amounts for each node.
            capacity: Vehicle capacity.
            R: Parameter R.
            C: Parameter C.
            params: HGS parameters.
        """
        super().__init__(dist_matrix, waste, capacity, R, C, params)

    def optimize(self, solution: "Individual") -> "Individual":
        """
        Iteratively improve an individual using local search operators.
        """
        if not solution.routes:
            return solution

        self.routes = [r[:] for r in solution.routes]

        # Run optimization with current profit context
        self._optimize_internal(initial_profit=solution.profit_score)

        solution.routes = self.routes
        solution.profit_score = self.current_profit
        new_gt = []
        for r in self.routes:
            new_gt.extend(r)

        # Preserve all nodes in giant_tour for genetic consistency (OX crossover)
        # Append any nodes that are NOT in the current routes to the end of giant_tour
        visited_set = set(new_gt)
        missing_nodes = [node for node in solution.giant_tour if node not in visited_set]
        new_gt.extend(missing_nodes)

        solution.giant_tour = new_gt
        return solution
