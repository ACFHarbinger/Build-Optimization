"""
RTS (Reactive Tabu Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.rts import RTSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import RTSParams
from .solver import RTSSolver


@RouteConstructorRegistry.register("rts")
class RTSPolicy(BaseRoutingPolicy):
    """Reactive Tabu Search policy class."""

    def __init__(self, config: Optional[Union[RTSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RTSConfig

    def _get_config_key(self) -> str:
        return "rts"

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
        Execute the Reactive Tabu Search (RTS) solver logic.

        RTS is an advanced variant of Tabu Search that automatically adjusts
        the tabu tenure (memory length) based on the search performance. It:
        - Increases the tenure if it detects that the search is cycling back to
          previously visited solutions.
        - Decreases the tenure if the search is not improving and no cycles
          are detected (encouraging intensification).
        - Triggers an escape mechanism (perturbation) if long-term search
          cycles are detected.
        This "reactive" feedback loop eliminates the need for manual parameter
        tuning of the tabu list length.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                RTS parameters (initial_tenure, tenure_increase, max_iterations).
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
        params = RTSParams(
            initial_tenure=int(values.get("initial_tenure", 7)),
            min_tenure=int(values.get("min_tenure", 3)),
            max_tenure=int(values.get("max_tenure", 20)),
            tenure_increase=float(values.get("tenure_increase", 1.5)),
            tenure_decrease=float(values.get("tenure_decrease", 0.9)),
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = RTSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
