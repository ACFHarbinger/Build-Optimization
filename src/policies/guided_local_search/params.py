"""
Configuration parameters for the Guided Local Search (GLS) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GLSParams:
    """
    Configuration for the GLS solver.

    GLS augments the objective function with adaptive penalty terms on
    edge features present at local optima.  The penalty makes previously
    visited optima less attractive, forcing the search into new basins.

    Attributes:
        lambda_param: Scaling coefficient for the penalty term.
        max_restarts: Number of GLS restart cycles.
        n_removal: Nodes removed per LLH destroy step.
        n_llh: Number of LLHs in the pool.
        inner_iterations: LLH iterations per GLS cycle.
        time_limit: Wall-clock time limit in seconds.
    """

    lambda_param: float = 0.3
    max_restarts: int = 50
    n_removal: int = 2
    n_llh: int = 5
    inner_iterations: int = 20
    time_limit: float = 60.0
