"""
GLS (Guided Local Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.gls import GLSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import GLSParams
from .solver import GLSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("gls")
class GLSPolicy(BaseRoutingPolicy):
    """Guided Large Neighborhood Search (G-LNS) policy class."""

    def __init__(self, config: Optional[Union[GLSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return GLSConfig

    def _get_config_key(self) -> str:
        return "gls"

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
        Execute the Guided Local Search (GLS) metaheuristic solver logic.

        GLS is a metaheuristic that sits on top of a local search algorithm. It
        augments the objective function with a penalty term for "bad" features
        (e.g., expensive edges) in the solution. When the local search reaches a
        local optimum, GLS identifies features contributing most to the high cost
        and increases their penalty, guiding the search out of the local
        optimum and towards unexplored areas of the search space.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                GLS parameters (lambda_param, alpha_param, penalty_cycles).
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
        params = GLSParams(
            lambda_param=float(values.get("lambda_param", 1.0)),
            alpha_param=float(values.get("alpha_param", 0.3)),
            penalty_cycles=int(values.get("penalty_cycles", 1000)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 6)),
            inner_iterations=int(values.get("inner_iterations", 100)),
            fls_coupling_prob=float(values.get("fls_coupling_prob", 0.8)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = GLSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
