"""
HGS-ALNS (Hybrid Genetic Search with ALNS Education) configuration for expert policy training.
"""

from dataclasses import dataclass


@dataclass
class HGSALNSConfig:
    """Configuration for Hybrid Genetic Search with ALNS Education (HGS-ALNS) expert policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        mu: Size of the genetic population.
        nb_elite: Number of elite individuals to preserve.
        mutation_rate: Probability of mutation.
        alns_education_iterations: Number of ALNS iterations for education phase.
        n_generations: Number of generations to evolve.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        engine: Solver engine to use.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    time_limit: float = 60.0
    mu: int = 50
    nb_elite: int = 10
    mutation_rate: float = 0.2
    alns_education_iterations: int = 50
    n_generations: int = 100
    max_vehicles: int = 0
    vrpp: bool = True
    crossover_rate: float = 0.8
    engine: str = "custom"
