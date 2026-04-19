"""
Progressive Hedging (PH) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class PHConfig:
    """Configuration for Progressive Hedging (PH) policy.

    Progressive Hedging (Rockafellar and Wets, 1991) is a horizontal
    decomposition algorithm for stochastic programming. It decomposes the
    stochastic VRP into scenario-specific subproblems and iteratively enforces
    non-anticipativity constraints.

    Attributes:
        rho: Penalty parameter for the quadratic non-anticipativity term.
            Larger values enforce consensus faster but may lead to suboptimality.
        max_iterations: Maximum number of PH iterations.
        convergence_tol: Tolerance for consensual visit decisions (non-anticipativity error).
        sub_solver: Key of the deterministic solver to use for subproblems (e.g., 'bc', 'alns').
        num_scenarios: Number of scenarios for SAA if none are provided.
        time_limit: Total wall-clock time limit in seconds.
        verbose: Enable detailed logging of PH iterations and convergence stats.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
    """

    rho: float = 1.0
    max_iterations: int = 50
    convergence_tol: float = 0.01
    sub_solver: str = "bc"
    num_scenarios: int = 10
    horizon: int = 7
    consensus_scope: str = "day_0"  # Options: "day_0", "full"
    stockout_penalty: float = 500.0
    time_limit: float = 300.0
    verbose: bool = True
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
