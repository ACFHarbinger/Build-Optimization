"""
Configuration parameters for the Guided Local Search (GLS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GLSParams:
    """
    Configuration for the GLS solver.

    GLS augments the objective function with adaptive penalty terms on
    edge features present at local optima.  The penalty makes previously
    visited optima less attractive, forcing the search into new basins.

    Attributes:
        lambda_param: Scaling coefficient for the penalty term (global intensity).
        alpha_param: Tuning parameter for the static base_lambda calculation.
        penalty_cycles: Number of GLS penalty update cycles (restarts).
        n_removal: Maximum nodes removed per LNS ruin step.
        n_llh: Number of Low-Level Heuristics in the pool.
        inner_iterations: Stagnation threshold representing Expected Neighborhood Coverage before declaring a local optimum and triggering a penalty update.
        fls_coupling_prob: Probability of triggering targeted penalized removal after update.
        time_limit: Wall-clock time limit in seconds.
    """

    lambda_param: float = 1.0
    alpha_param: float = 0.3
    penalty_cycles: int = 1000
    n_removal: int = 2
    n_llh: int = 6
    inner_iterations: int = 100
    fls_coupling_prob: float = 0.8
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
