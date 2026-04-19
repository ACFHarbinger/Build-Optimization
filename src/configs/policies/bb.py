"""
Branch-and-Bound (BB) configuration schema.

This module defines the dataclass used to configure the Branch-and-Bound policy,
mapping YAML configuration values to typed Python attributes for use in the solver.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class BBConfig:
    """Configuration for the Branch-and-Bound (BB) solver.

    The BB solver is an exact algorithm that explores a state-space search tree
    using Linear Programming (LP) relaxations to obtain lower bounds and prune
    branches that cannot yield a solution better than the current incumbent.

    **Formulation Selection**:
    The solver supports two mathematical formulations for subtour elimination:

    - **MTZ (Miller-Tucker-Zemlin)**: Compact formulation with O(n²) constraints.
      Uses load variables to prevent subtours. Custom B&B tree management with
      configurable branching strategies. Better for small-medium instances and
      when custom branching logic is desired.

    - **DFJ (Dantzig-Fulkerson-Johnson)**: Exponential constraints added lazily.
      Delegates B&B to Gurobi's internal engine. Better for large instances and
      when Gurobi license is available.

    Attributes:
        formulation (str): Mathematical formulation for subtour elimination.
            Supported: 'mtz', 'dfj', 'lr_uop'. Defaults to "dfj".

            - **"dfj"**: Dantzig-Fulkerson-Johnson with Gurobi's internal B&B engine
              and lazy subtour cuts. Best for large production instances.
            - **"mtz"**: Miller-Tucker-Zemlin compact formulation with a custom
              Python B&B tree. Best for small instances and ML-branching research.
            - **"lr_uop"**: Lagrangian Relaxation with an uncapacitated Orienteering
              Problem as the bounding oracle. Relaxes the capacity constraint with
              multiplier λ; tightens it via subgradient optimisation (Phase 1) before
              running a B&B tree (Phase 2). Best when capacity is the binding
              constraint and LP relaxations (MTZ/DFJ) are historically weak.

        time_limit (float): Maximum allowed runtime for the solver in seconds.
            Defaults to 60.0.
        mip_gap (float): Relative tolerance for the optimality gap. The solver
            terminates if (Incumbent - Bound) / Incumbent < mip_gap.
            Defaults to 0.01.
        branching_strategy (str): Logic for selecting the fractional variable to
            branch on. Only applies to MTZ formulation. Supported: 'most_fractional',
            'least_fractional', 'strong'. Defaults to "strong".
        strong_branching_limit (int): Maximum number of candidate variables to
            evaluate during strong branching. Only applies when branching_strategy='strong'.
            Defaults to 5.
        vrpp (bool): If True, enables the Vehicle Routing Problem with Profits
            mode, where all bins are candidates for collection based on profitability.
            Defaults to True.
        seed (Optional[int]): Random seed passed to the underlying Gurobi engine
            for deterministic results. Defaults to None.
        mandatory_selection (Optional[List[MandatorySelectionConfig]]): List of
            strategies determining mandatory nodes to visit. Defaults to None.
        route_improvement (Optional[List[RouteImprovingConfig]]): List of operations
            to refine the tour after the exact search completes. Defaults to None.

        # LR-UOP only
        lr_lambda_init (float): Initial Lagrange multiplier λ₀ ≥ 0 for Phase 1.
            Defaults to 0.0.
        lr_max_subgradient_iters (int): Maximum Polyak subgradient iterations.
            Defaults to 100.
        lr_subgradient_theta (float): Step-size multiplier θ ∈ (0, 2] in the
            Polyak rule α_k = θ·(L(λ_k) - LB_k) / ‖g_k‖². Defaults to 1.0.
        lr_subgradient_time_fraction (float): Fraction of time_limit devoted to
            Phase 1 (subgradient). Remainder is used for Phase 2 (B&B).
            Defaults to 0.4.
        lr_op_time_limit (float): Per-call time limit for each uncapacitated OP
            solve during both phases (seconds). Defaults to 10.0.
        lr_branching_strategy (str): Customer selection rule for B&B branching.
            - "max_waste": branch on the free OP-selected customer with the
              highest waste load (most aggressively reduces capacity violation).
            - "min_profit": branch on the free customer with the smallest
              modified Lagrangian profit (R - λ*)·w_i.
            Defaults to "max_waste".
        lr_max_bb_nodes (int): Maximum B&B nodes to explore in Phase 2 before
            stopping (safety backstop). Defaults to 5000.
    """

    formulation: str = "dfj"
    time_limit: float = 60.0
    mip_gap: float = 0.01
    branching_strategy: str = "strong"
    strong_branching_limit: int = 5
    vrpp: bool = True
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    # LR-UOP: Subgradient phase
    lr_lambda_init: float = 0.0
    lr_max_subgradient_iters: int = 100
    lr_subgradient_theta: float = 1.0
    lr_subgradient_time_fraction: float = 0.4

    # LR-UOP: Inner OP solver
    lr_op_time_limit: float = 10.0

    # LR-UOP: B&B phase
    lr_branching_strategy: str = "max_waste"
    lr_max_bb_nodes: int = 5000
