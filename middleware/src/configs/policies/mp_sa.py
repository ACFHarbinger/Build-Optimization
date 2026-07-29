"""
Configuration for the Multi-Period Simulated Annealing (MP-SA).
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_SA_Config:
    """
    Configuration for the Multi-Period Simulated Annealing (MP-SA) policy.

    Attributes:
        iters (int): Number of iterations. Defaults to 500.
        init_temp (float): Initial temperature. Defaults to 100.0.
        cooling_rate (float): Cooling rate. Defaults to 0.95.
        seed (int): Random seed for reproducibility. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 500
    init_temp: float = 100.0
    cooling_rate: float = 0.95
    seed: int = 42
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
