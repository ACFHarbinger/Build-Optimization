"""
Configuration for the Matheuristic Hyper-Heuristic (MHH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MHHConfig:
    """
    Configuration for the Matheuristic Hyper-Heuristic (MHH) policy.

    MHH applies matheuristic Low-Level Heuristics (LLHs) iteratively to
    find improvements in the multi-period plan.

    Attributes:
        iters (int): Number of iterations. Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 10
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
