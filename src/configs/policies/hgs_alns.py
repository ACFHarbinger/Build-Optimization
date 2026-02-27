"""
HGS-ALNS configuration.
"""

from dataclasses import dataclass


@dataclass
class HGSALNSConfig:
    engine: str = "hgs_alns"
    time_limit: float = 10.0
    population_size: int = 50
    elite_size: int = 10
    mutation_rate: float = 0.2
    crossover_rate: float = 0.8
    n_generations: int = 100
    alns_education_iterations: int = 50
