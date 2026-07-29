"""
RLS (Random Local Search) configuration.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RLSConfig:
    """Configuration for Random Local Search (RLS) policy.

    Attributes:
        n_iterations: Number of search iterations (operator applications).
        op_probs: Dictionary mapping operator names to selection probabilities.
                 Supported keys: 'two_opt', 'swap', 'relocate', 'two_opt_star', 'swap_star', 'three_opt'.
                 Normalized internally if they don't sum to 1.
        time_limit: Maximum time in seconds for the solver.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_iterations: int = 100
    op_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }
    )
    time_limit: float = 30.0
