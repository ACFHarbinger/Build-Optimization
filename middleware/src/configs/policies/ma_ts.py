"""
Stochastic Tournament GA configuration.

Replaces LCA with rigorous GA terminology.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MemeticAlgorithmToleranceBasedSelectionConfig:
    """Configuration for Memetic Algorithm Tolerance-based Selection (MA-TS) policy.

    GA with sigmoid-based pairwise tournament selection.
    """

    population_size: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    recombination_rate: float = 0.6
    perturbation_strength: int = 2
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
