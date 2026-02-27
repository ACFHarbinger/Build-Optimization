"""
Optimization configuration.
"""

from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """Optimization base configuration.

    Attributes:
        budget: Maximum cost/budget for the build.
        character_level: Character level constraint for items.
        time_limit: Maximum wall-clock time limit for the solver in seconds.
    """

    budget: float = 1000.0
    character_level: int = 50
    time_limit: float = 60.0
