"""
HH-ACO (Hyper-Heuristic Ant Colony Optimization) configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class HyperHeuristicACOConfig:
    """Configuration for Hyper-Heuristic Ant Colony Optimization (HH-ACO).

    Attributes:
        n_ants: Number of ants in the colony.
        alpha: Pheromone importance factor.
        beta: Heuristic (distance) importance factor.
        rho: Pheromone evaporation rate.
        tau_0: Initial pheromone level.
        tau_min: Minimum pheromone bound.
        tau_max: Maximum pheromone bound.
        max_iterations: Maximum number of ACO iterations.
        time_limit: Maximum time in seconds for the solver.
        q0: Probability of exploitation vs exploration (ACS-style).
        sequence_length: Length of operator sequence for hyper-heuristic ACO.
        local_search: Whether to apply local search after construction.
        local_search_iterations: Number of local search iterations.
        elitist_weight: Weight for elitist pheromone update.
        operators: List of local search operators to use.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: float = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    max_iterations: int = 50
    time_limit: float = 30.0
    seed: Optional[int] = None
    q0: float = 0.9
    sequence_length: int = 5
    local_search: bool = True
    local_search_iterations: int = 500
    elitist_weight: float = 1.0
    operators: List[str] = field(default_factory=lambda: ["swap", "2opt_intra", "relocate", "swap_star", "perturb"])
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
