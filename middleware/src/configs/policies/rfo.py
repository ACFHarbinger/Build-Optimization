"""
Configuration for the Relax-and-Fix-and-Optimize (RFO) matheuristic.
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class RFOConfig:
    """
    Configuration for the Relax-and-Fix-and-Optimize (RFO) policy.

    RFO decomposes the planning horizon into overlapping windows, solving
    successive sub-MIPs where early periods are fixed or solved as integer,
    and later periods are relaxed.

    Attributes:
        window_size (int): Size of the rolling integer window. Defaults to 3.
        step_size (int): Number of days to slide the window forward after each solve. Defaults to 2.
        mip_time (float): Time limit for each sub-MIP solve in seconds. Defaults to 60.0.
        mip_gap (float): Target optimality gap for sub-MIP solves. Defaults to 0.01.
        vrpp (bool): Whether the problem is a VRP with Profits. Defaults to True.
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    window_size: int = 3
    step_size: int = 2
    mip_time: float = 60.0
    mip_gap: float = 0.01
    vrpp: bool = True

    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
