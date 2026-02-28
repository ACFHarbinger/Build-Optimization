"""
Branch-Cut-and-Price (BCP) solver dispatcher for Build Optimization.

In the discrete build domain, this dispatches to ILP solvers (Gurobi or OR-Tools)
to solve the Multiple Choice Knapsack Problem (MCKP) optimally.
"""

from typing import Any, Dict, Tuple

import numpy as np

from core.problem import BuildProblem

from .gurobi_engine import run_bcp_gurobi
from .ortools_engine import run_bcp_ortools


def run_bcp(
    problem: BuildProblem,
    budget: float,
    values: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[np.ndarray, float]:
    """
    Main dispatcher for BCP / ILP solvers.

    Args:
        problem: BuildProblem instance.
        budget: Maximum cost budget.
        values: Configuration with 'bcp_engine'.

    Returns:
        Tuple[np.ndarray, float]: (best_build, best_score).
    """
    engine = values.get("bcp_engine", "ortools")

    if engine == "gurobi":
        return run_bcp_gurobi(problem, budget, values, **kwargs)
    else:
        # Default to OR-Tools
        return run_bcp_ortools(problem, budget, values, **kwargs)
