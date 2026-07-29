"""
Local Search Package.

This package contains implementations of local search algorithms used to refine
initial solutions. It includes a base class (`LocalSearch`) and specific
implementations for ACO and HGS.

Attributes:
    ACOLocalSearch (class): Local search for Ant Colony Optimization.
    HGSLocalSearch (class): Local search for Hybrid Genetic Search.
    LocalSearch (class): Abstract base class for local search.

Example:
    >>> from logic.src.policies.local_search import HGSLocalSearch
    >>> ls = HGSLocalSearch(dist_matrix, waste, capacity, R, C, params)
    >>> optimized_solution = ls.optimize(solution)
"""

from .local_search_aco import ACOLocalSearch
from .local_search_base import LocalSearch
from .local_search_hgs import HGSLocalSearch

__all__ = ["ACOLocalSearch", "HGSLocalSearch", "LocalSearch"]
