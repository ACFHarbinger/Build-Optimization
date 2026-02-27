"""
K-Sparse ACO Runner Module.

This module provides a high-level interface to run the K-Sparse ACO solver.
It handles parameter parsing from a configuration dictionary and initializes
the solver.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization.k_sparse_aco.runner import run_k_sparse_aco
    >>> result = run_k_sparse_aco(dist_matrix, wastes, capacity, ...)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .params import ACOParams
from .solver import KSparseACOSolver


def run_k_sparse_aco(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
    *args: Any,
) -> Tuple[List[List[int]], float, float]:
    """
    Main entry point for K-Sparse ACO solver.

    Args:
        dist_matrix: Distance matrix.
        wastes: Node wastes dictionary.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Configuration dictionary with ACO parameters.
        mandatory_nodes: List of mandatory node indices.
        *args: Additional arguments (ignored).

    Returns:
        Tuple[List[List[int]], float, float]: (routes, profit, cost)
    """
    params = ACOParams(
        n_ants=values.get("n_ants", 10),
        k_sparse=values.get("k_sparse", 15),
        alpha=values.get("alpha", 1.0),
        beta=values.get("beta", 2.0),
        rho=values.get("rho", 0.1),
        q0=values.get("q0", 0.9),
        tau_0=values.get("tau_0"),
        tau_min=values.get("tau_min", 0.001),
        tau_max=values.get("tau_max", 10.0),
        max_iterations=values.get("max_iterations", 100),
        time_limit=values.get("time_limit", 30.0),
        local_search=values.get("local_search", True),
        elitist_weight=values.get("elitist_weight", 1.0),
    )

    solver = KSparseACOSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
    return solver.solve()
