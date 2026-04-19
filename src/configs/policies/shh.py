"""
Configuration for the Selection Hyper-Heuristic (SHH).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class SHHConfig:
    """
    Configuration for the Selection Hyper-Heuristic (SHH) policy.

    SHH adaptively selects a Low-Level Heuristic (LLH) based on past
    performance and applies it to the current solution. It uses Late
    Acceptance Hill Climbing (LAHC) for acceptance.

    Attributes:
        iters (int): Number of iterations. Defaults to 200.
        history_len (int): Length of the Late Acceptance history. Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 200
    history_len: int = 10
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
