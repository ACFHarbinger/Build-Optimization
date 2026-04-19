"""
Kernel Search matheuristic configuration.

This module defines the settings used by the Kernel Search framework to decompose
and solve VRPP instances.
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class KernelSearchConfig:
    """
    Configuration for the Kernel Search matheuristic framework.

    Kernel Search (Angelelli et al., 2010) is a matheuristic designed to solve
    large-scale combinatorial optimization problems by restricting the binary
    variable space to a high-quality subset (the Kernel) and iteratively
    exploring additional candidate variables (Buckets) via restricted MIPs.

    Attributes:
        time_limit (float): Maximum allowed execution time for the entire matheuristic.
            If the cumulative runtime exceeds this value, the current iteration
            finishes and returns the best solution found so far. Defaults to 300.0s.
        initial_kernel_size (int): The number of decision variables (nodes and edges)
            included in the initial 'Kernel'. These are selected based on their
            fractional values in the initial LP relaxation. Defaults to 50.
        bucket_size (int): The number of variables added from the remaining set to
            the search space in each iterative sub-MIP. Small buckets lead to faster
            iterations but may miss synergies. Defaults to 20.
        max_buckets (int): The maximum number of buckets to process. Setting this
            restricts the depth of the search and total runtime. Defaults to 10.
        mip_limit_nodes (int): The branch-and-bound node limit for each sub-MIP
            solve. Prevents any single iteration from stalling the entire search.
            Defaults to 5000 nodes.
        mip_gap (float): The targeted relative optimality gap for each sub-problem.
            Defaults to 0.01 (1%).
        seed (int): Random seed for Gurobi's internal heuristics and algorithmic
            variations to ensure deterministic behavior. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits.
        engine (str): Identifier for the optimization engine. Use "gurobi" to
            invoke the project's native Gurobi-based KS solver.
        framework (str): Identifier for the optimization framework. Options
            include "ortools" and "pyomon".
        mandatory_selection (Optional[MandatorySelectionConfig]): Logic for pre-selecting
            mandatory bins (e.g., those nearly full or overdue).
        route_improvement (Optional[RouteImprovingConfig]): Settings for applying
            localized refinement (e.g., 2-opt, Or-opt) to the final matheuristic tour.
    """

    time_limit: float = 300.0
    initial_kernel_size: int = 50
    bucket_size: int = 20
    max_buckets: int = 10
    mip_limit_nodes: int = 5000
    mip_gap: float = 0.01
    seed: int = 42
    vrpp: bool = True

    # Infrastructure
    engine: str = "gurobi"
    framework: str = "ortools"
    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
