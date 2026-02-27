"""
LCA (League Championship Algorithm) configuration.
"""

from dataclasses import dataclass


@dataclass
class LCAConfig:
    engine: str = "lca"
    n_teams: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    crossover_prob: float = 0.6
    n_removal: int = 2
    time_limit: float = 60.0
