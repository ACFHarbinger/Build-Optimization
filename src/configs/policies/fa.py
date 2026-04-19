"""
FA (Discrete Firefly Algorithm) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class FAConfig:
    """
    Configuration for the Discrete Firefly Algorithm policy.

    Attributes:
        pop_size: Number of fireflies.
        beta0: Maximum attractiveness coefficient.
        gamma: Light absorption coefficient.
        alpha_profit: Profit weight in favourability score.
        beta_will: Willingness weight in favourability score.
        gamma_cost: Insertion-cost penalty weight in favourability score.
        alpha_rnd: Random-walk probability per firefly per iteration.
        max_iterations: Maximum iterations.
        n_removal: Number of nodes to remove in each iteration.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    pop_size: int = 20
    beta0: float = 1.0
    gamma: float = 0.1
    alpha_profit: float = 0.5
    beta_will: float = 0.3
    gamma_cost: float = 0.2
    alpha_rnd: float = 0.2
    max_iterations: int = 100
    n_removal: int = 3
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
