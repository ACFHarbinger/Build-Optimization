"""
SWC-TCF (Smart Waste Collection - Two-Commodity Flow) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class SWCTCFConfig:
    """Configuration for Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy.

    Attributes:
        Omega: Profit weight parameter.
        delta: Distance weight parameter.
        psi: Penalty parameter.
        time_limit: Maximum time in seconds for the solver.
        engine: Solver engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        framework: Solver framework to use ('ortools', 'pyomo').
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    Omega: float = 0.1
    delta: float = 0.0
    psi: float = 1.0
    time_limit: float = 600.0
    seed: Optional[int] = None
    engine: str = "gurobi"
    framework: str = "ortools"
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
