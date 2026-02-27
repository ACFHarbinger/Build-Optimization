"""
ALNS (Adaptive Large Neighborhood Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class ALNSConfig:
    engine: str = "alns"
    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    reaction_factor: float = 0.1
    min_removal: int = 1
    max_removal_pct: float = 0.3
