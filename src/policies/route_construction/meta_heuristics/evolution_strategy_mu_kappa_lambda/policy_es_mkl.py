"""
(μ,κ,λ) Evolution Strategy Policy Adapter with Age-Based Selection.

Adapts the (μ,κ,λ)-ES with age control for vehicle routing problems.
Individuals exceeding age κ are excluded from selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuKappaLambdaESConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import MuKappaLambdaESParams
from .solver import MuKappaLambdaESSolver


@RouteConstructorRegistry.register("es_mkl")
class MuKappaLambdaESPolicy(BaseRoutingPolicy):
    """
    (μ,κ,λ) Evolution Strategy policy with age-based selection.

    Age-limited parent survival prevents stagnation in long runs.
    """

    def __init__(self, config: Optional[Union[MuKappaLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ,κ,λ)-ES policy with optional config.

        Args:
            config: MuKappaLambdaESConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return MuKappaLambdaESConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "es_mkl"

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
        Execute the (mu, kappa, lambda) Evolution Strategy (ES) solver logic.

        (mu, kappa, lambda)-ES is a sophisticated evolutionary algorithm that
        introduces an "age" control parameter (kappa):
        - mu: The number of parent individuals.
        - lambda: The number of offspring generated.
        - kappa: The maximum age (in generations) an individual can survive
          multiple selection cycles.
        - Selection: Parents can survive across generations (like mu + lambda)
          but are strictly removed if they exceed age kappa. This prevents
          stagnation by ensuring constant population turnover while retaining
          beneficial elite individuals for a limited time.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                ES parameters (mu, kappa, lambda, sigma).
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
        tau_default: float = 1.0 / (2.0**0.5)
        params = MuKappaLambdaESParams(
            mu=values.get("mu", 10),
            kappa=values.get("kappa", 5),
            lambda_=values.get("lambda_", 5),
            rho=values.get("rho", 2),
            tau_local=values.get("tau_local", tau_default),
            tau_global=values.get("tau_global", tau_default),
            initial_sigma=values.get("initial_sigma", 1.0),
            recombination_type=values.get("recombination_type", "intermediate"),
            max_iterations=values.get("max_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
            min_sigma=values.get("min_sigma", 1e-10),
            max_sigma=values.get("max_sigma", 10.0),
            bounds_min=values.get("bounds_min", -5.0),
            bounds_max=values.get("bounds_max", 5.0),
            n_removal=values.get("n_removal", 3),
            stagnation_limit=values.get("stagnation_limit", 10),
            local_search_iterations=values.get("local_search_iterations", 100),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        # Initialize the generalized self-adaptive solver
        solver = MuKappaLambdaESSolver(
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
