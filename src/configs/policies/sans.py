"""
SANS (Simulated Annealing Neighborhood Search) configuration.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class SANSConfig:
    engine: Literal["new", "og"] = "new"
    T_init: float = 75.0
    iterations_per_T: int = 5000
    alpha: float = 0.95
    T_min: float = 0.01
    perc_bins_can_overflow: float = 0.0
    time_limit: float = 60.0
    combination: str = "2opt"
