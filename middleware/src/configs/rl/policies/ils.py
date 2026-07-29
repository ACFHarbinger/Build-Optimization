"""
ILS (Iterated Local Search) configuration for expert policy training.
"""

from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class ILSConfig:
    """Configuration for Iterated Local Search (ILS) expert policy.

    Attributes:
        n_restarts: Number of ILS restarts (perturbation cycles).
        ls_iterations: Iterations for local search within each phase.
        perturbation_strength: Fraction of tour to perturb.
        ls_operator: Local search operator name or dict of {name: prob}.
        perturbation_type: Perturbation method name or dict of {mode: prob}.
        time_limit: Maximum time in seconds for the solver.
        op_probs: Operator selection probabilities.
        perturb_probs: Perturbation method probabilities.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_restarts: int = 5
    ls_iterations: int = 50
    perturbation_strength: float = 0.2
    ls_operator: Union[str, Dict[str, float]] = "two_opt"
    perturbation_type: Union[str, Dict[str, float]] = "double_bridge"
    time_limit: float = 30.0
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
    perturb_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "double_bridge": 0.5,
            "shuffle": 0.3,
            "random_swap": 0.2,
        }
    )
