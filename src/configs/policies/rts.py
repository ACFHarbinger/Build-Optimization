"""
RTS (Reactive Tabu Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class RTSConfig:
    engine: str = "rts"
    initial_tenure: int = 7
    min_tenure: int = 3
    max_tenure: int = 20
    tenure_increase: float = 1.5
    tenure_decrease: float = 0.9
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
