"""
ABC Policy Adapter.

Adapts the Artificial Bee Colony (ABC) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.abc import ABCConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import ABCParams
from .solver import ABCSolver


@RouteConstructorRegistry.register("abc")
class ABCPolicy(BaseRoutingPolicy):
    """
    ABC policy class.

    Visits bins using the Artificial Bee Colony algorithm.
    """

    def __init__(self, config: Optional[Union[ABCConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ABCConfig

    def _get_config_key(self) -> str:
        return "abc"

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
        Execute the Artificial Bee Colony (ABC) solver logic.

        ABC is a population-based metaheuristic inspired by the foraging
        behavior of honey bees. It maintains a population of solutions
        (food sources), which are modified by three types of bees:
        - Employed Bees: Search around known food sources.
        - Onlooker Bees: Select high-quality food sources and search around
          them (exploitation).
        - Scout Bees: Abandon poor sources and find new random ones
          (exploration).
        The search process includes local search and removal operators for
        intensification.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ABC parameters (n_sources, limit, iterations).
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
        params = ABCParams(
            n_sources=int(values.get("n_sources", 20)),
            limit=int(values.get("limit", 10)),
            max_iterations=int(values.get("max_iterations", 200)),
            n_removal=int(values.get("n_removal", 1)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = ABCSolver(
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
