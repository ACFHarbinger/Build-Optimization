"""
Configuration for Local Branching with Variable Neighborhood Search (LB-VNS).
"""

from dataclasses import dataclass, field
from typing import Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class LocalBranchingVNSConfig:
    """
    Configuration for the Local Branching with Variable Neighborhood Search (LB-VNS) matheuristic.

    LB-VNS (Hansen, Mladenović, and Urošević, 2006) is a powerful hybrid that
    integrates the metaheuristic structure of Variable Neighborhood Search (VNS)
    with the tactical 'local' search capabilities of Local Branching (LB).

    The algorithm operates by systematically exploring neighborhoods of increasing
    size (defined by the Hamming distance 'k') whenever the current neighborhood
    fails to yield an improved solution. This allows the solver to escape local
    optima while maintaining the mathematical rigor of a MILP-based intensification.

    Attributes:
        time_limit (float): Total wall-clock time limit for the entire VNS process.
            If exceeded, the algorithm terminates and returns the best found solution.

        k_min (int): Initial neighborhood size (Hamming distance). This determines
             the 'radius' of the first restricted search space. Standard values range from 10 to 20.

        k_max (int): Maximum neighborhood size for the VNS loop. If LB-VNS reaches this
            size without finding an improvement, it may reset or terminate depending on
            the implementation. It acts as an upper bound on diversification intensity.

        k_step (int): Increment value for neighborhood expansion. Whenever a sub-problem
            is solved to optimality (or a limit) without finding a better solution,
            k is increased by this amount: k = k + k_step.

        time_limit_per_lb (float): Maximum time budget allocated for each internal
            Local Branching execution (local search phase). This ensures the solver
            doesn't get stuck too long in a single, potentially difficult, sub-MIP.

        max_lb_iterations (int): Limit on the number of internal improvement steps
            taken within each Local Branching call. In LB-VNS, the LB part itself
            can iterate to find a local optimum before the VNS moves to the next neighborhood.

        mip_gap (float): Relative optimality gap targeted for each sub-problem solve.
            A value like 0.01 (1%) allows the solver to stop once it's close enough to
            the local optimal solution, saving time for more VNS explorations.

        seed (int): Global random seed for both the Shaking mechanism (random perturbation)
            and the underlying Gurobi solver, ensuring deterministic and reproducible results.

        vrpp (bool): Whether the problem is a VRP with Profits.

        # Infrastructure Settings
        # -----------------------
        engine (str): Solver engine to use ('gurobi', 'scip', 'highs', or 'cplex').

        framework (str): Solver framework to use ('ortools', 'pyomo').

        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for identifying
            and enforcing mandatory nodes in the routing problem.

        route_improvement (Optional[RouteImprovingConfig]): Optional local search or
            heuristics (e.g., node swaps, 2-opt) to be applied to solutions found
            by the matheuristic for further minor refinement.
    """

    time_limit: float = 300.0
    k_min: int = 10
    k_max: int = 60
    k_step: int = 10
    time_limit_per_lb: float = 30.0
    max_lb_iterations: int = 15
    mip_gap: float = 0.01
    seed: int = 42
    vrpp: bool = True

    # Infrastructure
    engine: str = "gurobi"
    framework: str = "ortools"
    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    # Injected Acceptance Criterion
    acceptance: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="only_improving"))
