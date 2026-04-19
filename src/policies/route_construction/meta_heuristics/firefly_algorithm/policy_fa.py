"""
FA Policy Adapter.

Adapts the Discrete Firefly Algorithm (FA) solver to the agnostic
BaseRoutingPolicy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.fa import FAConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import FAParams
from .solver import FASolver


@RouteConstructorRegistry.register("fa")
class FAPolicy(BaseRoutingPolicy):
    """
    FA policy class.

    Visits bins using the Discrete Firefly Algorithm.
    """

    def __init__(self, config: Optional[Union[FAConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return FAConfig

    def _get_config_key(self) -> str:
        return "fa"

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
        Execute the Discrete Firefly Algorithm (FA) solver logic.

        FA is a nature-inspired metaheuristic based on the flashing behavior
        of fireflies. In this discrete version for the VRPP:
        - Attraction: Fireflies (solutions) are attracted to others with
          higher "brightness" (better profit).
        - Movement: Attraction triggers a movement where the less bright
          firefly shifts its structure towards the brighter one using
          probabilistic edge swaps and removals.
        - Light Intensity: Diminishes with distance, modeled through the
          gamma parameter to balance exploration and intensification.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                FA parameters (pop_size, beta0, gamma, alpha_profit).
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
        params = FAParams(
            pop_size=int(values.get("pop_size", 20)),
            beta0=float(values.get("beta0", 1.0)),
            gamma=float(values.get("gamma", 0.1)),
            alpha_profit=float(values.get("alpha_profit", 0.5)),
            beta_will=float(values.get("beta_will", 0.3)),
            gamma_cost=float(values.get("gamma_cost", 0.2)),
            alpha_rnd=float(values.get("alpha_rnd", 0.2)),
            n_removal=int(values.get("n_removal", 3)),
            max_iterations=int(values.get("max_iterations", 100)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = FASolver(
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
