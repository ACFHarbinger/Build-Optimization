"""
BP (Branch-and-Price) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class BPConfig:
    """Configuration for Branch-and-Price (BP) policy.

    Branch-and-Price is a column generation method that handles large-scale
    optimization by implicitly enumerating exponentially many variables (routes).

    The algorithm uses:
    1. Set partitioning master problem (select routes to cover nodes)
    2. Pricing subproblem (generate profitable routes via RCSPP)
    3. Column generation (iteratively add routes with positive reduced cost)
    4. Branch-and-bound for integrality (Ryan-Foster branching)

    Key advantages:
    - Handles problems with huge solution spaces
    - Provides strong LP bounds (tighter than compact formulations)
    - Scalable to large instances

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum column generation iterations per node.
        max_routes_per_iteration: Maximum routes to generate per pricing call.
        optimality_gap: Convergence tolerance for column generation.
        use_ryan_foster_branching: Use Ryan-Foster branching (not yet implemented).
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 300.0
    max_iterations: int = 100
    max_routes_per_iteration: int = 10
    optimality_gap: float = 1e-4
    use_ryan_foster_branching: bool = False
    multi_day_mode: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
