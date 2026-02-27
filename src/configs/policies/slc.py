"""
SLC (Soccer League Competition) configuration.
"""

from dataclasses import dataclass


@dataclass
class SLCConfig:
    engine: str = "slc"
    n_teams: int = 5
    team_size: int = 4
    max_iterations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    time_limit: float = 60.0
