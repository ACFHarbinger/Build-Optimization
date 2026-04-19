"""
Pure Island Model GA configuration.

Replaces SLC with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MemeticAlgorithmIslandModelConfig:
    """Configuration for Memetic Algorithm Island Model (MA-IM) policy.

    Functionally equivalent to SLC.
    """

    n_islands: int = 5
    island_size: int = 4
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
