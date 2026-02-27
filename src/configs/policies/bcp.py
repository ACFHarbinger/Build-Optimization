"""
BCP configuration.
"""

from dataclasses import dataclass


@dataclass
class BCPConfig:
    engine: str = "ortools"
    time_limit: float = 60.0
