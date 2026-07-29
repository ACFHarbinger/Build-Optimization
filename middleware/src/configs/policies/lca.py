"""
LCA (League Championship Algorithm) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class LCAConfig:
    """
    Configuration for the League Championship Algorithm policy.

    Attributes:
        n_teams: Number of teams in the championship.
        max_iterations: Maximum number of weeks (outer loop).
        tolerance_pct: Infeasibility tolerance fraction for match decisions.
        crossover_prob: Probability of OX crossover vs. perturbation after loss.
        n_removal: Nodes removed per perturbation step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    n_teams: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    crossover_prob: float = 0.6
    n_removal: int = 2
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
