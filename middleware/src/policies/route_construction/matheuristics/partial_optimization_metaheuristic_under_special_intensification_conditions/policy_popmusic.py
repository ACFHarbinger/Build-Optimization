"""
Simulator adapter for the POPMUSIC matheuristic.
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.popmusic import POPMUSICConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver import (
    run_popmusic,
)

from .params import POPMUSICParams


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.DECOMPOSITION,
    PolicyTag.META_HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("popmusic")
class POPMUSICPolicy(BaseRoutingPolicy):
    """
    Adapter for the POPMUSIC (Partial Optimization Metaheuristic Under Special
    Intensification Conditions) matheuristic.

    This policy encapsulates the LIFO-stack based neighborhood decomposition
    strategy defined by Taillard & Voss (2002). It systematically selects "seed"
    nodes, builds proximity-based subproblems (parts), and optimizes them using
    a base solver (typically LKH3 or Gurobi) until no further improvements are
    possible across all centroids.

    The adapter handles the conversion from the simulator's global state to the
    localized part-optimization context.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return POPMUSICConfig

    def _get_config_key(self) -> str:
        return "popmusic"

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
        """
         Execute the Partial Optimization Metaheuristic Under Special
         Intensification Conditions (POPMUSIC) solver logic.

         POPMUSIC is a decomposition-based matheuristic that breaks a large
        problem into smaller, overlapping sub-problems (parts) centered around
        "seed" nodes. Each part is optimized independently using a base solver
        (e.g., LKH-3 or Gurobi). If an improvement is found, the part is
        re-merged into the global solution, and the search continues until
        local optimality is achieved across all potential centroids.

         Args:
             sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                 sub-problem nodes.
             sub_wastes (Dict[int, float]): Mapping of local node indices to their
                 current bin inventory levels.
             capacity (float): Maximum vehicle collection capacity.
             revenue (float): Revenue obtained per kilogram of waste collected.
             cost_unit (float): Monetary cost incurred per kilometer traveled.
             values (Dict[str, Any]): Merged configuration dictionary containing
                 POPMUSIC settings, sub-problem sizes, and base solver choices.
             mandatory_nodes (List[int]): Local indices of bins that MUST be
                 collected in this period.
             **kwargs: Additional context, including:
                 - coords (np.ndarray): Spatial coordinates for clustering.
                 - search_context (Optional[SearchContext]): Context for tracking
                   recursive solver statistics.
                 - multi_day_context (Optional[MultiDayContext]): Context for
                   inter-day state propagation.

         Returns:
             Tuple[List[List[int]], float, float]: A 3-tuple containing:
                 - routes: Optimized collection routes (list-of-lists, local indices).
                 - profit: Total calculated net profit (Total Revenue - Total Cost).
                 - cost: Total travel cost calculated by the solver.
        """
        # 1. Initialize parameters
        params = POPMUSICParams.from_config(self.config)

        # 2. Enforce deterministic behavior (override seed if provided in kwargs)
        seed = kwargs.get("seed", params.seed)

        # 3. Call the core matheuristic solver with granular parameter extraction
        routes, total_routing_cost, profit, info = run_popmusic(
            coords=kwargs["coords"],
            mandatory=mandatory_nodes,
            distance_matrix=sub_dist_matrix,
            n_vehicles=kwargs.get("n_vehicles", 1),
            subproblem_size=params.subproblem_size,
            max_iterations=params.max_iterations,
            base_solver=params.base_solver,
            base_solver_config=params.base_solver_config,
            cluster_solver=params.cluster_solver,
            cluster_solver_config=params.cluster_solver_config,
            initial_solver=params.initial_solver,
            seed=seed,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            vrpp=params.vrpp,
            profit_aware_operators=params.profit_aware_operators,
            k_prox=params.k_prox,
            seed_strategy=params.seed_strategy,
        )

        return routes, profit, total_routing_cost
