"""
SA (Simulated Annealing) configuration.
"""

from dataclasses import dataclass


@dataclass
class SAConfig:
    engine: str = "sa"
    initial_temp: float = 100.0
    alpha: float = 0.995
    min_temp: float = 0.01
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
