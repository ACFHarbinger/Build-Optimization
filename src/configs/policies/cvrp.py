"""
CVRP configuration.
"""

from dataclasses import dataclass


@dataclass
class CVRPConfig:
    engine: str = "ortools"
    time_limit: float = 60.0
    cache: bool = False
