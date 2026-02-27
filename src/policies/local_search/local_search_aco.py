"""
ACO Local Search Module.

This module provides the local search implementation specifically for
Ant Colony Optimization (ACO). It focuses on 2-opt refinement.

Attributes:
    None

Example:
    >>> from logic.src.policies.local_search.local_search_aco import ACOLocalSearch
    >>> ls = ACOLocalSearch(dist_matrix, waste, capacity, R, C, params)
    >>> optimized_routes = ls.optimize(routes)
"""

from typing import List

from .local_search_base import LocalSearch


class ACOLocalSearch(LocalSearch):
    """
    Local Search module for K-Sparse ACO.
    Implements 2-opt local search refinement.
    """

    def optimize(self, solution: List[List[int]]) -> List[List[int]]:
        """
        Apply full set of local search operators to the solution.

        Args:
            solution: List of routes, where each route is a list of node indices.

        Returns:
            List of optimized routes.
        """
        # Create a deep copy of routes to modify
        self.routes = [r[:] for r in solution]

        # Run optimization
        self._optimize_internal()

        return self.routes
