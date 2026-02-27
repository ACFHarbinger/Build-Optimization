"""
HS (Harmony Search) configuration.
"""

from dataclasses import dataclass


@dataclass
class HSConfig:
    engine: str = "hs"
    hm_size: int = 10
    HMCR: float = 0.9
    PAR: float = 0.3
    max_iterations: int = 500
    time_limit: float = 60.0
