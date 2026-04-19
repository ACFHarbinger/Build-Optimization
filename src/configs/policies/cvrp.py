"""
CVRP (Capacitated Vehicle Routing Problem) policy configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CVRPConfig:
    """Configuration for Capacitated Vehicle Routing Problem (CVRP) policy.

    Attributes:
        cache: Whether to cache solutions.
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
        engine: Solver engine to use ('ortools', 'gurobi').
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    cache: bool = False
    time_limit: float = 60.0
    engine: str = "ortools"
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
