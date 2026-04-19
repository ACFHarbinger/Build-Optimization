"""
Configuration for the Column Generation Heuristic (CGH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CGHConfig:
    """
    Configuration for the Column Generation Heuristic (CGH) policy.

    CGH uses a master problem (set packing/partitioning) and generates routes
    heuristically. Instead of solving the exact subproblem (ESPPRC), it uses
    randomized greedy and local search to find negative reduced-cost columns.

    Attributes:
        cg_iters (int): Number of column generation iterations. Defaults to 10.
        routes_per_iter (int): Number of heuristic routes to generate in each
            column generation iteration. Defaults to 50.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    cg_iters: int = 10
    routes_per_iter: int = 50
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
