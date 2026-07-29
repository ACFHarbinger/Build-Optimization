"""
SISR (Slack Induction by String Removal) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class SISRConfig:
    """Configuration for Slack Induction by String Removal (SISR) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of SISR iterations.
        start_temp: Initial temperature for simulated annealing acceptance.
        cooling_rate: Rate at which temperature decreases each iteration.
        max_string_len: Maximum length of string to remove.
        avg_string_len: Average length of strings to remove.
        blink_rate: Probability of 'blinking' during insertion.
        destroy_ratio: Fraction of solution to destroy per iteration.
        vrpp: Whether this is a VRPP problem.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    time_limit: float = 10.0
    seed: Optional[int] = None
    max_iterations: int = 1000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    max_string_len: int = 10
    avg_string_len: float = 3.0
    blink_rate: float = 0.01
    destroy_ratio: float = 0.2
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
