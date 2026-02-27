"""
RRT (Record-to-Record Travel) configuration.
"""

from dataclasses import dataclass


@dataclass
class RRTConfig:
    engine: str = "rrt"
    tolerance: float = 0.05
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
