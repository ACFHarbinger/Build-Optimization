"""
LCA Policy Adapter.

Adapts the League Championship Algorithm (LCA) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.lca import LCAConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import LCAParams
from .solver import LCASolver


@RouteConstructorRegistry.register("lca")
class LCAPolicy(BaseRoutingPolicy):
    """
    LCA policy class.

    Visits bins using the League Championship Algorithm.
    """

    def __init__(self, config: Optional[Union[LCAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return LCAConfig

    def _get_config_key(self) -> str:
        return "lca"

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
        Execute the League Championship Algorithm (LCA) solver logic.

        LCA is a sports-inspired metaheuristic that mimics the competitive environment
        of a sports league. In this implementation:
        - Teams: A population of solutions representing "teams".
        - Matches: Pairs of teams play matches (solution comparisons). The team
          with the higher brightness (profit) is the winner.
        - Tactics: Winning tactics are propagated through the population via
          probabilistic movements and crossover, while losing teams adapt by
          exploring new search areas.
        This provides a structured framework for balancing intensification (among
        winners) and exploration (among losers).

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                LCA parameters (n_teams, tolerance_pct, crossover_prob).
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
        params = LCAParams(
            n_teams=int(values.get("n_teams", 10)),
            max_iterations=int(values.get("max_iterations", 100)),
            tolerance_pct=float(values.get("tolerance_pct", 0.05)),
            crossover_prob=float(values.get("crossover_prob", 0.6)),
            n_removal=int(values.get("n_removal", 2)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = LCASolver(
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
