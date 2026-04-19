"""
Configuration for the Adaptive Memory Programming Hyper-Heuristic (AMP-HH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class AMPHHConfig:
    """
    Configuration for the Adaptive Memory Programming Hyper-Heuristic (AMP-HH) policy.

    AMP-HH maintains a memory of high-quality components and reconstructs
    new solutions by combining these components using Low-Level Heuristics (LLHs).

    Attributes:
        mem_size (int): Size of the adaptive memory. Defaults to 10.
        iters (int): Number of iterations. Defaults to 50.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    mem_size: int = 10
    iters: int = 50
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
