"""
Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST) Configuration.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MemeticAlgorithmDualPopulationConfig:
    """
    Configuration for the Memetic Algorithm Dual Population policy.
    """

    # MADP Parameters
    population_size: int = 30
    max_iterations: int = 200
    diversity_injection_rate: float = 0.2
    elite_learning_weights: Optional[List[float]] = None
    elite_count: int = 3

    # Operators
    local_search_iterations: int = 500
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Common policy fields
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
