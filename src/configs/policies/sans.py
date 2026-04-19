"""
SANS (Simulated Annealing Neighborhood Search) configuration.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class SANSConfig:
    """Configuration for Simulated Annealing Neighborhood Search (SANS) policy.

    Attributes:
        engine: Engine type - 'new' for improved SA, 'og' for original LAC.
        time_limit: Maximum time in seconds for the solver.
        perc_bins_can_overflow: Percentage of bins allowed to overflow.
        T_min: Minimum temperature for simulated annealing.
        T_init: Initial temperature for simulated annealing.
        iterations_per_T: Number of iterations at each temperature level.
        alpha: Cooling rate (temperature multiplier).
        combination: LAC combination type ('a' or 'b') for 'og' engine.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    engine: Literal["new", "og"] = "new"
    time_limit: float = 60.0
    seed: Optional[int] = None
    perc_bins_can_overflow: float = 0.0
    T_min: float = 0.01
    T_init: float = 75.0
    iterations_per_T: int = 5000
    alpha: float = 0.95
    combination: Optional[Literal["a", "b"]] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
