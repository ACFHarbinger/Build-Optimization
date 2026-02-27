"""
VNS (Variable Neighborhood Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class VNSConfig:
    engine: str = "vns"
    k_max: int = 5
    max_iterations: int = 200
    local_search_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
