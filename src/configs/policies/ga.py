"""
GA (Genetic Algorithm) configuration.
"""

from dataclasses import dataclass


@dataclass
class GAConfig:
    engine: str = "ga"
    pop_size: int = 30
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    n_removal: int = 2
    time_limit: float = 60.0
