"""
ABC (Artificial Bee Colony) configuration.
"""

from dataclasses import dataclass


@dataclass
class ABCConfig:
    engine: str = "abc"
    n_sources: int = 20
    limit: int = 10
    max_iterations: int = 200
    n_removal: int = 1
    time_limit: float = 60.0
