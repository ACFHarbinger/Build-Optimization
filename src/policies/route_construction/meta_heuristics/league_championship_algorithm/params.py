"""
Configuration parameters for the League Championship Algorithm (LCA) solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LCAParams:
    """
    Configuration parameters for the LCA solver.

    Teams (routing solutions) compete in weekly matches.  After each match,
    the losing team analyses the winner's routing topology and produces a new
    formation via crossover or perturbation.  A controlled infeasibility
    tolerance allows mildly infeasible but structurally rich solutions to
    remain in contention, providing topological bridging between disconnected
    feasible regions.

    Attributes:
        n_teams: Number of teams in the championship.
        max_iterations: Maximum number of weeks (outer loop).
        tolerance_pct: Infeasibility tolerance as a fraction of best profit
                       (teams within this band may still win a match).
        crossover_prob: Probability of OX crossover vs. perturbation after loss.
        n_removal: Nodes removed per perturbation step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
    """

    n_teams: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    crossover_prob: float = 0.6
    n_removal: int = 2
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "LCAParams":
        """Create parameters from a configuration object."""
        return cls(
            n_teams=getattr(config, "n_teams", 10),
            max_iterations=getattr(config, "max_iterations", 100),
            tolerance_pct=getattr(config, "tolerance_pct", 0.05),
            crossover_prob=getattr(config, "crossover_prob", 0.6),
            n_removal=getattr(config, "n_removal", 2),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
