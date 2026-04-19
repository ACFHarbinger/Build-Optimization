"""
SISR Policy Adapter.

Adapts the Slack Induction by String Removal (SISR) logic to the common policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import SISRConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import SISRParams
from .solver import SISRSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LARGE_NEIGHBORHOOD_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("sisr")
class SISRPolicy(BaseRoutingPolicy):
    """
    Policy adapter for the SISR metaheuristic.
    """

    def __init__(self, config: Optional[Union[SISRConfig, Dict[str, Any]]] = None):
        """Initialize SISR policy with optional config.

        Args:
            config: SISRConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return SISRConfig

    def _get_config_key(self) -> str:
        return "sisr"

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
        Execute the Slack Induction by String Removal (SISR) solver logic.

        SISR is a state-of-the-art Large Neighborhood Search (LNS) variant that
        refines solutions by removing "strings" (sequences of contiguous nodes)
        from existing routes. This "string removal" operator creates significant
        "slack" in the route timing and capacity, which is then re-optimized
        by a greedy-with-blink insertion heuristic. The search is typically
        guided by a Simulated Annealing schedule, allowing it to explore far
        beyond local optima by periodically accepting destructive moves.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                SISR parameters (max_string_len, avg_string_len, blink_rate).
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
        params = SISRParams(
            time_limit=values.get("time_limit", 10.0),
            max_iterations=values.get("max_iterations", 1000),
            start_temp=values.get("start_temp", 100.0),
            cooling_rate=values.get("cooling_rate", 0.995),
            max_string_len=values.get("max_string_len", 10),
            avg_string_len=values.get("avg_string_len", 3.0),
            blink_rate=values.get("blink_rate", 0.01),
            destroy_ratio=values.get("destroy_ratio", 0.2),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SISRSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes=mandatory_nodes,
        )
        routes, profit, cost = solver.solve()
        return routes, profit, cost
