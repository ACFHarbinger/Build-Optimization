"""
PSOMA (Particle Swarm Optimization Memetic Algorithm) configuration for Hydra.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class PSOMAConfig:
    """
    Configuration for the PSOMA policy.

    Attributes:
        pop_size: Swarm size.
        omega: Inertia weight (low value ≈ 0.4 forces exploitation).
        c1: Cognitive acceleration coefficient.
        c2: Social acceleration coefficient.
        max_iterations: Maximum PSO iterations.
        local_search_freq: Memetic local-search frequency (every N iterations).
        n_removal: Nodes removed per local-search step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        mandatory_selection: Mandatory selection strategy config list.
        route_improvement: Route improvement operation config list.
    """

    pop_size: int = 20
    omega: float = 0.4
    c1: float = 1.5
    c2: float = 2.0
    max_iterations: int = 200
    local_search_freq: int = 10
    n_removal: int = 2
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
