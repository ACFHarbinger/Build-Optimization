"""
GPHH (Genetic Programming Hyper-Heuristic) configuration.
"""

from dataclasses import dataclass


@dataclass
class GPHHConfig:
    engine: str = "gphh"
    gp_pop_size: int = 20
    max_gp_generations: int = 30
    eval_steps: int = 50
    apply_steps: int = 200
    tree_depth: int = 3
    tournament_size: int = 3
    n_llh: int = 5
    n_removal: int = 2
    time_limit: float = 60.0
