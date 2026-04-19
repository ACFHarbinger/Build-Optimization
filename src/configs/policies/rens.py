"""
Relaxation Enforced Neighborhood Search (RENS) configuration.
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class RENSConfig:
    """
    Configuration for Relaxation Enforced Neighborhood Search (RENS) matheuristic.

    RENS (Berthold, 2009) is a start heuristic that explores the space of
    feasible roundings of an LP relaxation. It works by fixing all variables
    that take integer values in the LP relaxation and solving a restricted
    MIP on the remaining (fractional) variables.

    Attributes:
        time_limit (float): Total maximum runtime for the entire RENS process
            (including LP and MIP phases). Typical range is 30-120s.
        lp_time_limit (float): Hard limit for the initial continuous relaxation
            solve. LP relaxation is usually fast, so 5-10s is standard.
        mip_limit_nodes (int): Limit on branch-and-bound nodes for the sub-MIP.
            Used to prevent stalling on difficult neighborhoods.
        mip_gap (float): Optimality gap threshold for the sub-MIP. Setting this
            slightly higher (e.g., 0.05) often helps find a "good enough"
            primal solution quickly.
        seed (int): Random seed for the underlying Gurobi solver to ensure
            deterministic and reproducible results.
        vrpp (bool): Whether the problem is a VRP with Profits.
        framework: Optimization framework to use ('ortools' or 'pyomo').
        engine: Optimization engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        mandatory_selection (Optional[MandatorySelectionConfig]): Composition handle
            for bin selection strategies (e.g., collecting bins at risk of overflow).
        route_improvement (Optional[RouteImprovingConfig]): Settings for
            post-optimization refinement (e.g., Local Search or Route-Smoothing).
    """

    time_limit: float = 60.0
    lp_time_limit: float = 10.0
    mip_limit_nodes: int = 1000
    mip_gap: float = 0.01
    seed: int = 42
    vrpp: bool = True

    # Infrastructure
    framework: str = "ortools"
    engine: str = "gurobi"
    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
