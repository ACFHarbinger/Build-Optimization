"""
Simulator adapter for the Adaptive Kernel Search (AKS) matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.aks import AdaptiveKernelSearchConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .aks import run_adaptive_kernel_search_gurobi
from .params import AKSParams


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("aks")
class AdaptiveKernelSearchPolicy(BaseRoutingPolicy):
    """
    Simulator adapter for the Adaptive Kernel Search (AKS) matheuristic.

    Adaptive Kernel Search (Guastaroba et al., 2017) is an advanced matheuristic
    framework designed to solve complex Mixed-Integer Linear Programming (MILP)
    problems by decomposing them into a sequence of adaptive, more tractable
    restricted sub-problems.

    Relationship to Basic Kernel Search:
    While basic Kernel Search relies on a static decomposition into a fixed
    Kernel and steady-sized buckets, AKS introduces a dynamic, self-tuning
    loop that reacts to objective improvements by expanding the search
    neighborhood (buckets) and permanently promoting variables to the Kernel.

    Algorithm Summary:
    1.  **Selection Phase**: Computes the MILP root node relaxation of the VRPP model,
        including fractional subtour separation, to obtain accurate profitability
        scores for all variables.
    2.  **Kernel Solve / GETFEASIBLE**: Solves an initial MIP on the 'Kernel'.
        If infeasible, it iteratively expands the Kernel until a solution is found.
    3.  **Difficulty Assessment**: Classifies the instance as EASY, NORMAL, or HARD
        based on the initial MIP solve time.
    4.  **Adaptive Improvement Phase**:
        - **EASY**: Rapidly adds chunks of variables until a time threshold is reached.
        - **HARD**: Permanently fixes variables with extremely high relaxation values
          to shrink the search space.
        - **NORMAL**: Performs a rigorous pass through partitioned buckets.
    5.  **Final Solve**: Performs a final optimization over the promoted Kernel.

    Attributes:
        config (Optional[Dict[str, Any]]): The raw Hydra configuration dictionary.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Adaptive Kernel Search policy.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Return the configuration dataclass.
        """
        return AdaptiveKernelSearchConfig

    def _get_config_key(self) -> str:
        """
        Return the unique Hydra configuration key "aks".
        """
        return "aks"

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
        """Not used - AKS requires specialized execute()."""
        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[List[int], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """
        Execute the Adaptive Kernel Search (AKS) matheuristic on the simulation state.

        AKS iteratively decomposes the VRPP into a set of tractable sub-problems
        by selecting a subset of promising variables (the Kernel). It then
        adaptively expands this Kernel by solving MIP buckets, promoting variables
        that belong to high-quality feasible solutions discovered during the
        search.

        This implementation uses Gurobi as the underlying MIP solver and
        includes subtour elimination constraints.

        Args:
            **kwargs: Context for matheuristic execution, including:
                - distance_matrix (np.ndarray): Full distance matrix for the problem.
                - wastes (Dict[int, float]): Bin inventory levels.
                - capacity (float): Vehicle collection capacity.
                - mandatory (List[int]): Local indices of mandatory bins.
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
        # 1. Initialize parameters
        params = AKSParams.from_config(self.config)

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

        # 5. Call the core standalone AKS solver with granular parameter extraction
        tour, obj_val, cost = run_adaptive_kernel_search_gurobi(
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
            t_easy=params.t_easy,
            epsilon=params.epsilon,
            seed=seed,
        )

        # 6. Return standardized results
        return tour, cost, obj_val, kwargs.get("search_context"), kwargs.get("multi_day_context")
