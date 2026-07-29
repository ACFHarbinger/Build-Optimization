"""
Configuration for the Lagrangian Relaxation Heuristic (LRH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class LRHConfig:
    """
    Configuration for the Lagrangian Relaxation Heuristic (LRH) policy.

    LRH relaxes the capacity constraints using Lagrangian multipliers and
    solves a simplified problem in each iteration. It uses a subgradient
    method to update the multipliers and find dual bounds.

    Attributes:
        max_iter (int): Maximum number of subgradient iterations. Defaults to 50.
        step_size (float): Initial step size for subgradient method. Defaults to 2.0.
        halving_freq (int): Frequency (iterations) at which to halve the step size.
            Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    max_iter: int = 50
    step_size: float = 2.0
    halving_freq: int = 10
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
