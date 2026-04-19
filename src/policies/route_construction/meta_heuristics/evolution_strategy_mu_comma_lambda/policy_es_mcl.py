"""
(μ,λ) Evolution Strategy Policy Adapter.

Adapts the rigorous (μ,λ)-ES implementation into the overarching policy registry.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuCommaLambdaESConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import MuCommaLambdaESParams
from .solver import MuCommaLambdaESSolver


@RouteConstructorRegistry.register("es_mcl")
class MuCommaLambdaESPolicy(BaseRoutingPolicy):
    """
    (μ,λ) Evolution Strategy policy class.

    Executes a strict generational evolutionary algorithm with truncation selection.
    """

    def __init__(self, config: Optional[Union[MuCommaLambdaESConfig, Dict[str, Any]]] = None):
        """
        Initialize (μ,λ)-ES policy with optional config.

        Args:
            config: MuCommaLambdaESConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MuCommaLambdaESConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "es_mcl"

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
        Execute the (mu, lambda) Evolution Strategy (ES) solver logic.

        (mu, lambda)-ES is a rigorous generational evolutionary algorithm:
        - mu: The number of parents selected to produce the next generation.
        - lambda: The number of offspring generated from the parents.
        - सिलेक्शन (Selection): Only the lambda offspring are considered for
          the next generation (the parents are discarded), using truncation
          selection (best mu out of lambda).
        This implementation applies discrete mutation (node swaps/removals)
        and optional local search to refine offspring.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ES parameters (mu, lambda, max_iterations).
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
        # Map configuration dictionary to strict ES parameters
        params = MuCommaLambdaESParams(
            mu=values.get("mu", 15),
            lambda_=values.get("lambda_", 100),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 500),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = MuCommaLambdaESSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost
