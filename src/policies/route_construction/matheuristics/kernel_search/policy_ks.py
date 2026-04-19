"""
Simulator adapter for the Kernel Search matheuristic.

This module provides the `KernelSearchPolicy` class, which acts as a bridge between the
WSmart+ Route simulator environment and the Gurobi-based Kernel Search solver logic.
It handles configuration parsing, state extraction, and results formatting.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.ks import KernelSearchConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.kernel_search.solver import run_kernel_search_gurobi

from .params import KSParams


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("ks")
class KernelSearchPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Kernel Search matheuristic.

    Kernel Search (Angelelli et al., 2010) is a matheuristic framework designed to solve
    complex Mixed-Integer Linear Programming (MILP) problems by decomposing them into a
    sequence of smaller, more tractable restricted sub-problems.

    The framework operates in two main phases:
    1.  **Selection Phase**: An initial 'Kernel' of decision variables is identified, typically
        using the fractional values from an LP relaxation or other indicators of profitability.
    2.  **Improvement Phase**: The remaining variables are partitioned into 'Buckets'. Sub-MIPs
        are solved iteratively by adding variables from one or more buckets to the current kernel,
        updating the kernel with "useful" variables (those that take positive values in the solution).

    This implementation is specialized for the Vehicle Routing Problem with Profits (VRPP),
    leveraging Gurobi to solve the underlying mathematical model.

    Attributes:
        config (Dict[str, Any]): The raw Hydra configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Kernel Search policy with a specific configuration.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary-like configuration object,
                typically provided by Hydra. If None, default settings are used.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the configuration dataclass associated with this policy.

        This allows the simulator to perform type checking and validation on the
        provided configuration parameters.

        Returns:
            Type: The `KernelSearchConfig` class.
        """
        return KernelSearchConfig

    def _get_config_key(self) -> str:
        """
        Return the unique Hydra configuration key for Kernel Search.

        This key is used to look up the algorithm-specific parameters in the
        Hydra configuration hierarchy.

        Returns:
            str: The configuration key "ks".
        """
        return "ks"

    def _run_solver(
        self,
        sub_dist_matrix: Any,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Not used - KS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Kernel Search (KS) matheuristic on the simulation state.

        Kernel Search decomposes the MILP model of a VRPP into a sequence of
        smaller sub-problems. It starts with an initial "Kernel" of variables
        (selected via LP relaxation) and iteratively adds "Buckets" of
        remaining variables, solving a series of MIPs to improve the current
        best feasible solution.

        This implementation uses Gurobi for the MIP solves and manages
        subtour elimination dynamically.

        Args:
            **kwargs: Context for matheuristic execution, including:
                - distance_matrix (np.ndarray): Symmetric or asymmetric cost matrix.
                - wastes (Dict[int, float]): Map of customer IDs to current waste levels.
                - capacity (float): Maximum vehicle load.
                - mandatory (List[int]): IDs of customers that MUST be visited.
                - R (float): Revenue multiplier for the objective function.
                - C (float): Cost multiplier for the objective function.
                - seed (int): Random seed for reproducibility.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple of:
                - tour: The ordered sequence of visited nodes, starting/ending at 0.
                - cost: The total travel distance/cost of the resulting tour.
                - obj_val: The final net profit (Revenue - Cost).
                - Optional[SearchContext]: Updated search context.
                - Optional[MultiDayContext]: Updated multi-period context.
        """
        # 1. Initialize parameters
        params = KSParams.from_config(self.config)

        # 2. Extract environment parameters from the simulation state
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("mandatory", [])

        # 3. Handle objective multipliers (Revenue and Cost weights)
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)

        # 4. Enforce deterministic behavior (override seed if provided in kwargs)
        seed = kwargs.get("seed", params.seed)

        # 5. Call the core matheuristic solver with granular parameter extraction
        tour, obj_val, cost = run_kernel_search_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            initial_kernel_size=params.initial_kernel_size,
            bucket_size=params.bucket_size,
            max_buckets=params.max_buckets,
            time_limit=params.time_limit,
            mip_limit_nodes=params.mip_limit_nodes,
            mip_gap=params.mip_gap,
            seed=seed,
        )

        # 6. Return standard policy results
        return tour, cost, obj_val, kwargs.get("search_context"), kwargs.get("multi_day_context")
