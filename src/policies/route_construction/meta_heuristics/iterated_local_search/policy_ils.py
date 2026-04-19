"""
ILS (Iterated Local Search) Policy Adapter.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ils import ILSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import ILSParams
from .solver import ILSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("ils")
class ILSPolicy(BaseRoutingPolicy):
    """Iterated Local Search policy class."""

    def __init__(self, config: Optional[Union[ILSConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return ILSConfig

    def _get_config_key(self) -> str:
        return "ils"

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
        Execute the Iterated Local Search (ILS) metaheuristic solver logic.

        ILS is a simple yet powerful metaheuristic that iteratively applies a
        local search to a perturbed solution. It operates in a loop:
        - Perturbation: The current solution is modified (e.g., via ruin-and-recreate
          or random moves) to escape the current local optimum.
        - Local Search: The perturbed solution is refined using local search
          operators until a new local optimum is found.
        - Acceptance: The new local optimum is accepted or rejected based on
          the objective function.
        In this implementation, multiple restarts and inner iterations are
        governed by the configuration parameters.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ILS parameters (n_restarts, perturbation_strength, inner_iterations).
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
        params = ILSParams(
            n_restarts=int(values.get("n_restarts", 30)),
            inner_iterations=int(values.get("inner_iterations", 20)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            perturbation_strength=float(values.get("perturbation_strength", 0.15)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=int(values.get("seed", 42)),
            vrpp=bool(values.get("vrpp", True)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
        )

        solver = ILSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()
