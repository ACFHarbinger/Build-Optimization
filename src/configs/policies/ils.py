"""
ILS (Iterated Local Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class ILSConfig:
    engine: str = "ils"
    n_restarts: int = 30
    inner_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    perturbation_strength: float = 0.15
    time_limit: float = 60.0
