"""
K-Sparse ACO Runner Module for Build Optimization.
"""

from typing import Any, Dict, Tuple

import numpy as np

from core.problem import BuildProblem

from .params import ACOParams
from .solver import KSparseACOSolver


def run_k_sparse_aco(
    problem: BuildProblem,
    budget: float,
    values: Dict[str, Any],
    *args: Any,
) -> Tuple[np.ndarray, float]:
    """
    Main entry point for K-Sparse ACO solver.

    Args:
        problem: BuildProblem instance.
        budget: Maximum cost allowed.
        values: Configuration dictionary with ACO parameters.
        *args: Additional arguments (ignored).

    Returns:
        Tuple[np.ndarray, float]: (best_build, best_score)
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

    solver = KSparseACOSolver(problem, budget, params)
    return solver.solve()
