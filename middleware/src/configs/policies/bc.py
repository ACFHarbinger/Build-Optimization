"""
BC (Branch-and-Cut) configuration.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class BCConfig:
    """Configuration for Branch-and-Cut (BC) policy.

    Branch-and-Cut is an exact optimization method that combines:
    - Cutting planes to strengthen LP relaxations
    - Branch-and-bound for integer optimization
    - Gurobi solver for MIP solving

    The algorithm solves the VRPP to provable optimality using:
    1. Initial LP relaxation
    2. Separation algorithms for violated inequalities:
       - Subtour elimination constraints (SEC)
       - Capacity inequalities
       - Comb inequalities (optional)
    3. Branch-and-bound when cuts are insufficient

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        mip_gap: Relative MIP optimality gap (0.0 = prove optimality).
        use_heuristics: Whether to use primal heuristics for warm start.
        use_exact_separation: Use exact max-flow for SEC (slower but stronger).
        max_cuts_per_round: Maximum cuts to add per separation round.
        enable_fractional_capacity_cuts: Enable exact fractional RCC separation.
            True = Use O(V⁴) max-flow for fractional capacity cuts (small instances).
            False = Disable exact fractional capacity separation (large instances).
            Recommended: True for n ≤ 75, False for n > 75.
            Note: Automatically disabled for instances with n > 75 regardless of setting.
        profit_aware_operators: Whether to use profit-aware operators.
        vrpp: Whether to use VRPP expand pool.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
        use_saa: Whether to use Sample Average Approximation (SAA) for SIRP.
        num_scenarios: Number of scenarios to generate for SAA.
    """

    time_limit: float = 300.0
    mip_gap: float = 0.0
    use_heuristics: bool = True
    use_exact_separation: bool = False
    max_cuts_per_round: int = 50
    enable_fractional_capacity_cuts: bool = True
    profit_aware_operators: bool = False
    vrpp: bool = False
    use_saa: bool = False
    num_scenarios: int = 10
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
