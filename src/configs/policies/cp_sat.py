from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CPSATConfig:
    """
    Configuration for the Constraint Programming (CP-SAT) policy.

    Attributes:
        num_days: Multi-day horizon.
        time_limit: Maximum solve time in seconds.
        search_workers: Number of parallel workers for CP-SAT.
        mip_gap: Relative gap for optimality (conceptually similar in CP-SAT).
        scaling_factor: Precision factor for integer conversion (default 1000).
        waste_weight: Weight for waste collected in the objective.
        cost_weight: Weight for travel cost in the objective.
        overflow_penalty: Penalty per unit of overflow.
        mean_increment: Expected daily waste increment for bins.
        seed: Random seed for reproducibility.
        mandatory_selection: Node selection strategy configuration.
        route_improvement: Route improvement operations.
    """

    num_days: int = 3
    time_limit: float = 300.0
    search_workers: int = 8
    mip_gap: float = 0.01
    scaling_factor: int = 1000

    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0
    mean_increment: float = 0.2

    seed: Optional[int] = None

    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
