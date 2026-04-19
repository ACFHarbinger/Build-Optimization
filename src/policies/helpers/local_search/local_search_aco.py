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

from typing import Any, Dict, List, Optional

import numpy as np

from .local_search_base import LocalSearch


class ACOLocalSearch(LocalSearch):
    """
    Local Search module for K-Sparse ACO.
    Implements 2-opt local search refinement.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        neighbors: Optional[Dict[int, List[int]]] = None,
    ):
        """Initialize ACO Local Search."""
        super().__init__(dist_matrix, waste, capacity, R, C, params, neighbors)

    def optimize(self, solution: List[List[int]], target_neighborhood: Optional[str] = None) -> List[List[int]]:
        """
        Apply local search operators to the solution.

        Args:
            solution: List of routes, where each route is a list of node indices.
            target_neighborhood: If provided, only applies the specified neighborhood operator.
                               If None or "all", applies all available operators.
                               Valid values: "intra_relocate", "intra_swap", "intra_2opt",
                               "intra_3opt", "intra_or_opt", "inter_relocate", "inter_swap",
                               "inter_2opt_star", "inter_swap_star", "cross_exchange",
                               "improved_cross_exchange", "lambda_interchange", "cyclic_transfer",
                               "exchange_chains", "ejection_chains", "relocate_chain", "three_permutation",
                               "unrouted_insert", "all"

        Returns:
            List of optimized routes.
        """
        # Create a deep copy of routes to modify
        self.routes = [r[:] for r in solution]

        # Run optimization with optional neighborhood filter
        self._optimize_internal(target_neighborhood=target_neighborhood)

        return self.routes
