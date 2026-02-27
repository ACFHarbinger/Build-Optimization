"""
SCA (Sine Cosine Algorithm) configuration.
"""

from dataclasses import dataclass


@dataclass
class SCAConfig:
    engine: str = "sca"
    pop_size: int = 20
    a_max: float = 2.0
    max_iterations: int = 200
    time_limit: float = 60.0
