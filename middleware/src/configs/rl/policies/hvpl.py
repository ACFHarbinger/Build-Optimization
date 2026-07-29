"""
HVPL (Hybrid Volleyball Premier League) configuration for expert policy training.
"""

from dataclasses import dataclass


@dataclass
class HVPLConfig:
    """Configuration for Hybrid Volleyball Premier League (HVPL) expert policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of league iterations.
        n_teams: Number of teams (solutions) per instance in the league.
        sub_rate: Rate of substitution for weak teams each iteration.
        aco_iterations: Number of ACO iterations for initialization/substitution.
    """

    time_limit: float = 60.0
    max_iterations: int = 50
    n_teams: int = 10
    sub_rate: float = 0.2
    aco_iterations: int = 1
