"""
AHVPL (Augmented Hybrid Volleyball Premier League) configuration for expert policy training.
"""

from dataclasses import dataclass


@dataclass
class AHVPLConfig:
    """Configuration for Augmented Hybrid Volleyball Premier League (AHVPL) expert policy.

    Extends HVPL with HGS integration parameters for diversity-driven
    crossover and bi-criteria fitness evaluation.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of league iterations.
        n_teams: Number of teams (solutions) per instance in the league.
        sub_rate: Rate of substitution for weak teams each iteration.
        aco_iterations: Number of ACO iterations for initialization/substitution.
        crossover_rate: Probability of applying OX crossover.
        nb_elite: Number of elite solutions for biased fitness ranking.
    """

    time_limit: float = 60.0
    max_iterations: int = 20
    n_teams: int = 10
    sub_rate: float = 0.2
    aco_iterations: int = 1
    crossover_rate: float = 0.7
    nb_elite: int = 5
