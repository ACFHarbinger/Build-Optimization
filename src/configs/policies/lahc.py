"""
LAHC (Late Acceptance Hill Climbing) configuration.
"""

from dataclasses import dataclass


@dataclass
class LAHCConfig:
    engine: str = "lahc"
    queue_size: int = 50
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
