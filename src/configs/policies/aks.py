"""
Configuration for the Adaptive Kernel Search (AKS) matheuristic.
"""

from dataclasses import dataclass
from typing import Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class AdaptiveKernelSearchConfig:
    """
    Configuration for the Adaptive Kernel Search (AKS) matheuristic framework.

    AKS (Guastaroba et al., 2017) is an advanced evolution of the Kernel Search
    matheuristic. While basic Kernel Search uses a static decomposition strategy,
    AKS introduces dynamic adjustment mechanisms that respond to the solver's
    performance during the iterative improvement phase.

    Key improvements over basic Kernel Search:
    1.  **Dynamic Bucket Sizing**: Adjusts the number of variables considered in
        each iteration based on whether an improvement was found. This allows
        the algorithm to "dig deeper" when it finds promising regions of the
        solution space.
    2.  **Variable Promotion**: Variables that contribute to a new best solution
        in a sub-MIP are permanently promoted to the Kernel. This ensures that
        high-quality building blocks are preserved and utilized in all
        subsequent sub-MIP solves.
    3.  **Adaptive Time Management**: Dynamically allocates the remaining
        wall-clock time budget to each sub-MIP based on the number of
        remaining buckets and the observed difficulty of recent solves.

    Attributes:
        time_limit (float): Maximum total wall-clock time (seconds) allowed for the
            entire AKS optimization process. Defaults to 300.0s.
        initial_kernel_size (int): The number of decision variables (nodes and edges)
            included in the initial 'Kernel' based on their LP relaxation scores.
            A larger kernel provides better initial solutions but increases the
            complexity of the first sub-MIP. Defaults to 50.
        bucket_size (int): The initial number of variables added in each iterative
            improvement sub-MIP. Defaults to 20.
        max_buckets (int): The maximum number of increments (sub-MIP iterations)
            to perform. This prevents the algorithm from running too long if many
            variables remain. Defaults to 15.
        mip_limit_nodes (int): The hard limit on branch-and-bound nodes for each
            individual sub-MIP execution. Prevents the solver from stalling on
            difficult sub-problems. Defaults to 10000.
        mip_gap (float): The relative optimality gap target for each sub-problem solve.
            A value of 0.01 means the solver stops when it is within 1% of the
            theoretical optimum for that restricted MIP. Defaults to 0.01.
        seed (int): Random seed for Gurobi's internal algorithms, ensuring
            computational reproducibility across different runs. Defaults to 42.
        vrpp (bool): Whether the problem is a VRP with Profits.

    # Adaptive Features (AKS)
    t_easy: float = 10.0
    epsilon: float = 0.1
    time_limit_stage_1: float = 0.2

        # Infrastructure
        engine (str): Identifier for the optimization engine. Use "gurobi" to
            invoke the project's native Gurobi-based AKS solver.
        framework (str): Identifier for the optimization framework. Options
            include "ortools" and "pyomon".
        mandatory_selection (Optional[MandatorySelectionConfig]): Configuration for
            mandatory node selection policies (e.g., must collect full bins).
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps applied after the main AKS loop.
    """

    time_limit: float = 300.0
    initial_kernel_size: int = 50
    bucket_size: int = 20
    max_buckets: int = 15
    mip_limit_nodes: int = 10000
    mip_gap: float = 0.01
    seed: int = 42
    vrpp: bool = True

    # Adaptive Features (AKS)
    t_easy: float = 10.0
    epsilon: float = 0.1
    time_limit_stage_1: float = 0.2

    # Infrastructure
    engine: str = "gurobi"
    framework: str = "ortools"
    mandatory_selection: Optional[MandatorySelectionConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
