"""
Configuration for the Multi-Period Iterated Local Search (MP-ILS).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_ILS_Config:
    """
    Configuration for the Multi-Period Iterated Local Search (MP-ILS) policy.

    Attributes:
        iters (int): Number of iterations. Defaults to 50.
        perturb_size (int): Number of nodes to remove during perturbation. Defaults to 3.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 50
    perturb_size: int = 3
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
