"""
Configuration parameters for Hybrid Genetic Search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logic.src.configs.policies import HGSConfig


@dataclass
class HGSParams:
    """
    Configuration parameters for Hybrid Genetic Search.

    Attributes:
        time_limit: Maximum search time in seconds.
        population_size: Target population size.
        elite_size: Number of elite individuals for survivor selection.
        mutation_rate: Probability of applying local search improvement.
        max_vehicles: Maximum number of vehicles allowed (0 = unlimited).
    """

    time_limit: float = 60.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    n_generations: int = 100
    max_vehicles: int = 0

    @classmethod
    def from_config(cls, config: HGSConfig) -> HGSParams:
        """Create HGSParams from a HGSConfig dataclass.

        Args:
            config: HGSConfig dataclass with solver parameters.

        Returns:
            HGSParams instance with values from config.
        """
        return cls(
            time_limit=config.time_limit,
            population_size=config.population_size,
            elite_size=config.elite_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            n_generations=config.n_generations,
            max_vehicles=config.max_vehicles,
        )
