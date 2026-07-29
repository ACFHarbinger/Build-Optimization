from dataclasses import dataclass
from typing import Optional


@dataclass
class LBBDConfig:
    """
    Configuration for the Logic-Based Benders Decomposition (LBBD) policy.
    """

    # Multi-day horizon
    num_days: int = 3
    # Whether to model multiple scenarios explicitly in the Master Problem
    stochastic_master: bool = False

    # Mean fractional increment per day (0.0 to 1.0)
    mean_increment: float = 0.2

    # Number of scenarios for SAA in Subproblem routing cost estimation (if used)
    # Note: LBBD typically assumes deterministic subproblems or expected costs.
    num_scenarios: int = 5

    # Algorithm parameters
    max_iterations: int = 20
    benders_gap: float = 0.01

    # Solve time limits
    time_limit: float = 300.0
    subproblem_timeout: float = 20.0
    mip_gap: float = 0.05

    # Objective weights
    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0

    # Whether to use Nogood cuts for feasibility
    use_nogood_cuts: bool = True
    # Whether to use logic-based optimality cuts for travel cost
    use_optimality_cuts: bool = True

    # Seed for reproducibility
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.num_days < 1:
            raise ValueError("num_days must be at least 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
