"""
Relaxation Enforced Neighborhood Search (RENS) policy adapter.

This module acts as the interface between the high-level WSmart-Route policy
framework and the low-level RENSSolver. It handles configuration mapping,
data extraction from the environment state, and orchestrates the solver's execution.

Strategy:
    The RENS policy is categorized as a Matheuristic. It leverages exact solvers
    (Gurobi) to solve a restricted neighborhood derived from an
    LP relaxation of the routing problem.

Reference:
    Berthold, T. (2009). "Rens - The relaxation enforced neighborhood search".
    ZIB-Report 09-11.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.rens import RENSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import RENSParams
from .solver import run_rens_gurobi


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("rens")
class RENSPolicy(BaseRoutingPolicy):
    """
    Policy adapter for the RENS (Relaxation Enforced Neighborhood Search) algorithm.

    This class prepares the environment data (distance matrices, bin states,
    vehicle capacities) and passes them to the RENSSolver for optimization.

    Deployment strategy:
        1. Context Extraction: Retrieves mandatory bins, current waste levels,
           and distances from the simulation orchestrator.
        2. Solver Invocation: Delegates the heavy lifting (MIP setup and solve)
           to the RENSSolver engine.
        3. Feature Mapping: Translates standard base class parameters (R, C,
           capacity) from the Hydra config into the solver's internal dictionary.
        4. Result Translation: Converts the tour sequence and objective value
           back into the simulation-standard metadata package.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RENS policy with a configuration dictionary.

        Args:
            config: Hydra configuration for RENS.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration dataclass associated with this policy."""
        return RENSConfig

    def _get_config_key(self) -> str:
        """Return the unique Hydra configuration key for RENS."""
        return "rens"

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
        """Not used - RENS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Relaxation Enforced Neighborhood Search (RENS) matheuristic
        on the simulation state.

        RENS is a simple but effective matheuristic that explores the
        neighborhood defined by the rounding of an LP relaxation solution.
        It solves a restricted MIP where variables are fixed or bounded based
        on their values in the initial LP solution, drastically reducing the
        search space.

        This implementation uses Gurobi for both the LP relaxation and the
        restricted MIP phase.

        Args:
            **kwargs: Context for matheuristic execution, including:
                - distance_matrix (np.ndarray): Cost matrix between nodes.
                - wastes (Dict[int, float]): Current waste levels for each bin.
                - capacity (float): Vehicle volume limit.
                - mandatory (List[int]): Optional nodes requiring collection.
                - R (float): Revenue multiplier.
                - C (float): Cost multiplier.
                - seed (int): Optional seed override.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple of:
                - tour: The optimized daily tour (starting/ending at depot 0).
                - cost: Total travel cost calculated by the solver.
                - obj_val: Final calculated net profit (Revenue - Cost).
                - Optional[SearchContext]: Updated search context.
                - Optional[MultiDayContext]: Updated multi-period context.
        """
        # 1. Initialize parameters
        params = RENSParams.from_config(self.config)

        # 2. Extract environment parameters
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("mandatory", [])

        # 3. Handle objective multipliers
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)

        # 4. Enforce deterministic behavior (override seed if provided in kwargs)
        seed = kwargs.get("seed", params.seed)

        # 5. Call the core matheuristic solver with granular parameter extraction
        tour, obj_val, cost = run_rens_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            time_limit=params.time_limit,
            lp_time_limit=params.lp_time_limit,
            mip_gap=params.mip_gap,
            seed=seed,
        )

        return tour, cost, obj_val, kwargs.get("search_context"), kwargs.get("multi_day_context")
