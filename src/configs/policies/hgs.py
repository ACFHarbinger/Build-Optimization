"""
HGS (Hybrid Genetic Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class HGSConfig:
    engine: str = "hgs"
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7
    n_generations: int = 100
    time_limit: float = 60.0
