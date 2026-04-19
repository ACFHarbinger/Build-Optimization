"""
VNS (Variable Neighborhood Search) Policy Adapter.

Adapts the VNS solver to the agnostic BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.vns import VNSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.variable_neighborhood_search.params import VNSParams
from logic.src.policies.route_construction.meta_heuristics.variable_neighborhood_search.solver import VNSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.NEIGHBORHOOD_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("vns")
class VNSPolicy(BaseRoutingPolicy):
    """
    Variable Neighborhood Search policy class.

    Solves the VRPP by systematically exploring a hierarchy of shaking
    neighborhoods (N_1 ... N_{k_max}) with a local search descent between
    each shaking step.  An improvement resets k to 1; exhausting all
    k_max structures completes one outer iteration.
    """

    def __init__(self, config: Optional[Union[VNSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return VNSConfig

    def _get_config_key(self) -> str:
        return "vns"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Execute the Variable Neighborhood Search (VNS) solver logic.

        VNS is a metaheuristic that systematically changes the neighborhood
        structure during the search to escape local optima. In this
        implementation:
        - Shaking: A random solution is generated in the k-th neighborhood of
           the current best solution (intensification/exploration tradeoff).
        - Local Search: A descent heuristic is applied to the shaken solution.
        - Neighborhood Change: If the refined solution is better, the search
          resets to the first neighborhood (N_1); otherwise, it moves to
          the next neighborhood (N_{k+1}).
        By cycling through neighborhoods of increasing size/complexity, VNS
        effectively explores different structural regions of the solution space.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                VNS parameters (k_max, max_iterations, local_search_iterations).
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
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
        params = VNSParams(
            k_max=int(values.get("k_max", 5)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = VNSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
