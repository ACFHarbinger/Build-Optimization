"""
Configuration for the Local Branching (LB) matheuristic.
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class LocalBranchingConfig:
    """
    Configuration for the Local Branching (LB) matheuristic.

    Local Branching (Fischetti and Lodi, 2003) is a matheuristic that uses a
    general-purpose MIP solver to explore the k-neighborhood of a given
    incumbent solution.

    Attributes:
        time_limit (float): Total wall-clock time limit for the policy.
        time_limit_per_iteration (float): Time limit for each sub-MIP solve.
        k (int): Neighborhood size (Hamming distance).
        max_iterations (int): Maximum number of improvement iterations.
        node_limit_per_iteration (int): Branch-and-bound node limit per sub-MIP.
        mip_gap (float): Targeted relative optimality gap for sub-problems.
        seed (int): Random seed for reproducibility.
        vrpp (bool): Whether the problem is a VRP with Profits.

        # Infrastructure
        engine (str): Solver engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        framework (str): Solver framework to use ('ortools', 'pyomo').
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for mandatory
            node selection.
        route_improvement (Optional[RouteImprovingConfig]): Optional local search refinement.
    """

    time_limit: float = 300.0
    time_limit_per_iteration: float = 30.0
    k: int = 10
    max_iterations: int = 20
    node_limit_per_iteration: int = 5000
    mip_gap: float = 0.01
    seed: int = 42
    vrpp: bool = True

    # Infrastructure
    engine: str = "gurobi"
    framework: str = "ortools"
    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
