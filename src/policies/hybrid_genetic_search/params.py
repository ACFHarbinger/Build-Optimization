"""
Configuration parameters for Hybrid Genetic Search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    pass


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
        crossover_rate: Probability of applying crossover.
        n_generations: Number of generations to run.
        n_offspring: Number of offspring to generate per generation.
    """

    time_limit: float = 60.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    n_generations: int = 100
    n_offspring: int = 20
    max_vehicles: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HGSParams:
        """Create HGSParams from a dictionary."""
        return cls(
            time_limit=float(data.get("time_limit", 60.0)),
            population_size=int(data.get("population_size", 50)),
            elite_size=int(data.get("elite_size", 10)),
            mutation_rate=float(data.get("mutation_rate", 0.2)),
            crossover_rate=float(data.get("crossover_rate", 0.7)),
            n_generations=int(data.get("n_generations", 100)),
            n_offspring=int(data.get("n_offspring", 20)),
            max_vehicles=int(data.get("max_vehicles", 0)),
        )
