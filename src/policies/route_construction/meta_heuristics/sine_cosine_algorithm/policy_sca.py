"""
SCA Policy Adapter.

Adapts the Sine Cosine Algorithm (SCA) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.sca import SCAConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.params import SCAParams
from logic.src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver import SCASolver


@RouteConstructorRegistry.register("sca")
class SCAPolicy(BaseRoutingPolicy):
    """
    SCA policy class.

    Visits bins using the Sine Cosine Algorithm.
    """

    def __init__(self, config: Optional[Union[SCAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SCAConfig

    def _get_config_key(self) -> str:
        return "sca"

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
        Execute the Sine Cosine Algorithm (SCA) solver logic.

        SCA is a population-based optimization algorithm that utilizes the
        properties of sine and cosine functions to explore and exploit the
        search space. It updates the position of candidate solutions (agents)
        based on their current positions and the best known solution, weighted
         by adaptive sine/cosine oscillations. This periodicity allows the
        algorithm to escape local optima effectively while the decaying
        amplitude ensures convergence towards the global optimum.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                SCA parameters (pop_size, a_max, max_iterations).
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
        params = SCAParams(
            pop_size=int(values.get("pop_size", 20)),
            a_max=float(values.get("a_max", 2.0)),
            max_iterations=int(values.get("max_iterations", 200)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SCASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
