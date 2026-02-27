"""
PSOMA (Particle Swarm Optimization Memetic Algorithm) configuration.
"""

from dataclasses import dataclass


@dataclass
class PSOMAConfig:
    engine: str = "psoma"
    pop_size: int = 20
    omega: float = 0.4
    c1: float = 1.5
    c2: float = 2.0
    max_iterations: int = 200
    local_search_freq: int = 10
    n_removal: int = 2
    time_limit: float = 60.0
