"""
Simulator adapter for the Local Branching with Variable Neighborhood Search (LB-VNS).

This policy adapts the LB-VNS matheuristic (Hansen et al., 2006) for use within
the WSmart+ Route simulation framework, extracting problem state and
orchestrating the Gurobi optimization process.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.lb_vns import LocalBranchingVNSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns import (
    run_lb_vns_gurobi,
)
from logic.src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params import (
    LBVNSParams,
)


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.NEIGHBORHOOD_SEARCH,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("lb_vns")
class LocalBranchingVNSPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Local Branching with Variable Neighborhood Search (LB-VNS).

    This policy implements the high-level integration required to run the LB-VNS
    matheuristic within the WSmart+ Route simulator. It handles:
    1. Configuration parsing and dataclass conversion.
    2. State extraction (distance matrices, node profits, capacities).
    3. Orchestration of the Local Branching intensification and VNS shaking phases.
    4. Solution reconstruction and coordinate-to-node mapping.

    Technical Context:
        LB-VNS is designed for hard combinatorial problems where standard local search
        fails to reach global minima. By systematically increasing the radius of the
        'restricted' MILP search space, it avoids entrapment in sub-optimal attraction basins.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LB-VNS policy with a provided or default configuration.

        Args:
            config: Optional configuration dictionary containing hyperparameters for
                the neighborhood sequence (k_min, k_max, k_step) and solver limits.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the structured configuration dataclass for LB-VNS.

        Returns:
            Type[LocalBranchingVNSConfig]: The dataclass defining LB-VNS hyperparameters.
        """
        return LocalBranchingVNSConfig

    def _get_config_key(self) -> str:
        """
        Return the top-level YAML key used for this policy's configuration.

        Returns:
            str: "lb_vns"
        """
        return "lb_vns"

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
        """Not used - LB-VNS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Local Branching with Variable Neighborhood Search (LB-VNS)
        matheuristic on the simulation state.

        LB-VNS combines the intensification power of Local Branching with the
        diversification capabilities of Variable Neighborhood Search. It
        iteratively explores K-neighborhoods around an incumbent solution and
        applies shaking (VNS) to escape local optima when Local Branching
        converges without further improvement.

        This implementation uses Gurobi for the restricted MIP solves.

        Args:
            **kwargs: Context for matheuristic execution, including:
                - distance_matrix (np.ndarray): Symmetric or asymmetric cost matrix.
                - wastes (Dict[int, float]): Map of customer IDs to collection profits.
                - capacity (float): Maximum weight the vehicle can carry.
                - mandatory (List[int]): IDs of nodes that must be visited.
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
                - obj_val: Final net profit (Revenue - Cost).
                - Optional[SearchContext]: Updated search context.
                - Optional[MultiDayContext]: Updated multi-period context.
        """
        # 1. Initialize parameters
        params = LBVNSParams.from_config(self.config)

        # 2. Extract environment parameters
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("mandatory", [])
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)

        # 3. Enforce deterministic behavior (override seed if provided in kwargs)
        seed = kwargs.get("seed", params.seed)

        # 4. Call the core matheuristic solver with curated parameters
        tour, obj_val, cost = run_lb_vns_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            mip_gap=params.mip_gap,
            seed=seed,
            params=params,
        )

        return tour, cost, obj_val, kwargs.get("search_context"), kwargs.get("multi_day_context")
