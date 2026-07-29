"""
ALNS (Adaptive Large Neighborhood Search) configuration for expert policy training.
"""

from dataclasses import dataclass


@dataclass
class ALNSConfig:
    """Configuration for Adaptive Large Neighborhood Search (ALNS) expert policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing acceptance.
        cooling_rate: Rate at which temperature decreases each iteration.
        reaction_factor: Weight update reaction factor for operator scores.
        min_removal: Minimum number of nodes to remove per destroy operation.
        max_removal_pct: Maximum percentage of nodes to remove per destroy operation.
        engine: Solver engine to use ('custom', 'alns').
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    max_removal_pct: float = 0.3
