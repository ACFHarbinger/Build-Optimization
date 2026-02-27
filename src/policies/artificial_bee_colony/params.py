"""
Configuration parameters for the Artificial Bee Colony (ABC) solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ABCParams:
    """
    Configuration parameters for the ABC solver.

    Mimics honey-bee foraging: each food source is a routing solution.
    Employed bees exploit their source; onlooker bees are directed by
    fitness-proportionate selection; scout bees reinitialise stagnant sources.

    Attributes:
        n_sources: Number of food sources (employed bees).
        limit: Trial counter threshold before a source is abandoned.
        max_iterations: Maximum ABC cycles.
        n_removal: Number of nodes removed per neighbourhood perturbation.
        time_limit: Wall-clock time limit in seconds.
    """

    n_sources: int = 20
    limit: int = 10
    max_iterations: int = 200
    n_removal: int = 1
    time_limit: float = 60.0
