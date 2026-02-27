"""
LKH (Lin-Kernighan-Helsgaun) configuration.
"""

from dataclasses import dataclass


@dataclass
class LKHConfig:
    engine: str = "lkh"
    check_capacity: bool = True
    max_iterations: int = 100
    time_limit: float = 60.0
