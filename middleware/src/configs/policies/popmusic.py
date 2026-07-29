"""
POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions) configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .alns import ALNSConfig
from .hgs import HGSConfig
from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig
from .tsp import TSPConfig


@dataclass
class POPMUSICSubSolverConfig:
    """Nested configuration for sub-solvers within POPMUSIC.

    Attributes:
        tsp: Configuration for TSP solver.
        alns: Configuration for ALNS solver.
        hgs: Configuration for HGS solver.
    """

    tsp: Optional[TSPConfig] = field(default_factory=TSPConfig)
    alns: Optional[ALNSConfig] = field(default_factory=ALNSConfig)
    hgs: Optional[HGSConfig] = field(default_factory=HGSConfig)


@dataclass
class POPMUSICConfig:
    """Configuration for POPMUSIC matheuristic.
    Based on Taillard & Voss (2002).

    Attributes:
        subproblem_size: Number of neighboring routes to optimize (R).
        max_iterations: Maximum number of sub-problem attempts without improvement.
        base_solver: The solver configuration used for subproblem optimization.
        base_solver_config: Detailed configuration for the subproblem solver.
        cluster_solver: The solver configuration used for clustering.
        cluster_solver_config: Detailed configuration for the clustering solver.
        initial_solver: Solver used to generate the initial solution (e.g., 'nearest_neighbor').
        seed: Random seed for reproducibility.
        vrpp: Whether this is a VRPP problem.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    # Core POPMUSIC parameters
    subproblem_size: int = 3
    k_prox: int = 10  # proximity network radius (paper §2.1)
    seed_strategy: str = "lifo"  # "lifo" | "fifo" | "random"
    max_iterations: Optional[int] = None  # None = run until U=empty

    # Solver orchestration
    base_solver: str = "alns"
    base_solver_config: Optional[POPMUSICSubSolverConfig] = field(default_factory=POPMUSICSubSolverConfig)
    cluster_solver: str = "fast_tsp"
    cluster_solver_config: Optional[POPMUSICSubSolverConfig] = field(default_factory=POPMUSICSubSolverConfig)
    initial_solver: str = "nearest_neighbor"
    seed: Optional[int] = None

    # Infrastructure
    vrpp: bool = True
    profit_aware_operators: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
