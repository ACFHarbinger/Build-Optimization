"""
Simulator adapter for the Local Branching (LB) matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.lb import LocalBranchingConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.local_branching.lb import run_local_branching_gurobi
from logic.src.policies.route_construction.matheuristics.local_branching.params import LBParams


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("lb")
class LocalBranchingPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Local Branching (LB) matheuristic.

    Local Branching (Fischetti and Lodi, 2003) is a matheuristic framework that
    systematically improves an incumbent solution by exploring its
    k-neighborhood using a standard MILP solver.

    Relationship to Other Matheuristics:
    Unlike Kernel Search which decomposes variables by their contribution to the
    LP relaxation, Local Branching decomposes the SOLUTION SPACE into small,
    tractable neighborhoods centered around the current best solution.

    Algorithm Summary:
    1.  **Phase 1: Initialization**: Find an initial feasible solution (e.g.,
        by solving the full VRPP model with a short time limit).
    2.  **Phase 2: Iterative Search**:
        - Define a neighborhood by adding a 'Local Branching Cut'
          (Hamming distance constraint).
        - Solve the resulting restricted MIP.
        - If an improvement is found, update the incumbent and jump to the
          new neighborhood.
    3.  **Phase 3: Termination**: Stop if no improvement is found within the
        current neighborhood, or if the global time/iteration budget is reached.

    Attributes:
        config (Optional[Dict[str, Any]]): The raw Hydra configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Local Branching policy with a specific configuration.

        Args:
            config (Optional[Dict[str, Any]]): A dictionary-like configuration object.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the configuration dataclass associated with this policy.

        Returns:
            Type: The `LocalBranchingConfig` class.
        """
        return LocalBranchingConfig

    def _get_config_key(self) -> str:
        """
        Return the unique Hydra configuration key for LB.

        Returns:
            str: The configuration key "lb".
        """
        return "lb"

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
        """Not used - LB requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Local Branching (LB) matheuristic on the simulation state.

        Local Branching explores the solution space by defining K-neighborhoods
        centered around an incumbent solution. It uses "Local Branching Cuts"
        (Hamming distance constraints) to restrict the MIP search space to
        these neighborhoods, allowing the solver to find improvements
        efficiently.

        This implementation uses Gurobi for the iterative neighborhood solves.

        Args:
            **kwargs: Context for matheuristic execution, including:
                - distance_matrix (np.ndarray): Symmetric or asymmetric cost matrix.
                - wastes (Dict[int, float]): Bin inventory levels.
                - capacity (float): Vehicle collection capacity.
                - mandatory (List[int]): Local indices of mandatory nodes.
                - R (float): Revenue multiplier.
                - C (float): Cost multiplier.
                - seed (int): Random seed for reproducibility.
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple of:
                - tour: The optimized daily tour (list of node IDs).
                - cost: Total travel cost calculated by the solver.
                - obj_val: Net calculated profit (Revenue - Cost).
                - Optional[SearchContext]: Updated search context.
                - Optional[MultiDayContext]: Updated multi-period context.
        """
        # 1. Initialize type-safe Params
        params = LBParams.from_config(self._config or kwargs.get("config", {}).get("lb", {}))

        # 2. Extract state parameters
        distance_matrix = kwargs["distance_matrix"]
        wastes = kwargs.get("wastes", {})
        capacity = kwargs.get("capacity", 1.0e9)
        mandatory_nodes = kwargs.get("mandatory", [])
        R = kwargs.get("R", 1.0)
        C = kwargs.get("C", 1.0)
        seed = params.seed if params.seed is not None else kwargs.get("seed", 42)

        # 3. Call the core LB solver
        tour, obj_val, cost = run_local_branching_gurobi(
            dist_matrix=distance_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            mandatory_nodes=mandatory_nodes,
            k=params.k,
            max_iterations=params.max_iterations,
            time_limit=params.time_limit,
            time_limit_per_iteration=params.time_limit_per_iteration,
            mip_limit_nodes=params.node_limit_per_iteration,
            mip_gap=params.mip_gap,
            seed=seed,
        )

        return tour, cost, obj_val, kwargs.get("search_context"), kwargs.get("multi_day_context")
