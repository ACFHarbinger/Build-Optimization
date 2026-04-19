"""
ALNS Policy Adapter.

Adapts the Adaptive Large Neighborhood Search (ALNS) logic to the agnostic interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import ALNSConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .dispatcher import run_alns


@RouteConstructorRegistry.register("alns")
class ALNSPolicy(BaseRoutingPolicy):
    """
    ALNS policy class.

    Visits pre-selected 'mandatory' bins using Adaptive Large Neighborhood Search.
    """

    def __init__(self, config: Optional[Union[ALNSConfig, Dict[str, Any]]] = None):
        """Initialize ALNS policy with optional config.

        Args:
            config: ALNSConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ALNSConfig

    def _get_config_key(self) -> str:
        """Return config key for ALNS."""
        return "alns"

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
        Execute the Adaptive Large Neighborhood Search (ALNS) solver logic.

        ALNS is a powerful metaheuristic that iteratively improves a solution by
        applying a sequence of destruction (removal) and reconstruction
        (insertion) operators. The selection of operators is controlled by a
        roulette-wheel mechanism that adapts based on the historical performance
        of each operator in discovering high-quality solutions.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ALNS parameters (iterations, cooling rate, operator weights).
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
        routes, profit, solver_cost = run_alns(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            values,
            mandatory_nodes,
            recorder=self._viz,
        )
        return routes, profit, solver_cost
