"""
Distance-Based PSO Policy Adapter.

Adapts the rigorous Distance-Based PSO implementation (replaces Firefly Algorithm).
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import DistancePSOConfig
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import DistancePSOParams
from .solver import DistancePSOSolver


@RouteConstructorRegistry.register("psoda")
class DistancePSOPolicy(BaseRoutingPolicy):
    """
    Distance-Based Particle Swarm Optimization policy class.

    PSO with exponential distance decay. Replaces Firefly Algorithm.
    """

    def __init__(self, config: Optional[Union[DistancePSOConfig, Dict[str, Any]]] = None):
        """Initialize Distance-Based PSO policy with optional config.

        Args:
            config: DistancePSOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return DistancePSOConfig

    def _get_config_key(self) -> str:
        """Return config key."""
        return "psoda"

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
        Execute the Distance-Based Particle Swarm Optimization (PSO-DA)
        solver logic.

        PSO-DA is a specialized version of PSO that incorporates an
        exponential distance decay for particle attraction, similar to the
        light intensity decay in Firefly Algorithms. This hybrid approach
        combines the velocity momentum and personal/global best tracking of
        PSO with the distance-dependent influence characteristic of bio-inspired
        models. It uses discrete mutation rates derived from particle velocities
        to maintain diversity and prevent premature convergence in discrete
        routing domains.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                PSO-DA parameters (population_size, inertia_weight, alpha_profit).
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
        params = DistancePSOParams(
            population_size=values.get("population_size", 20),
            max_iterations=values.get("max_iterations", 500),
            inertia_weight_start=values.get("inertia_weight_start", 0.9),
            inertia_weight_end=values.get("inertia_weight_end", 0.4),
            cognitive_coef=values.get("cognitive_coef", 2.0),
            social_coef=values.get("social_coef", 2.0),
            n_removal=values.get("n_removal", 3),
            velocity_to_mutation_rate=values.get("velocity_to_mutation_rate", 0.1),
            local_search_iterations=values.get("local_search_iterations", 100),
            time_limit=values.get("time_limit", 60.0),
            alpha_profit=values.get("alpha_profit", 1.0),
            beta_will=values.get("beta_will", 0.5),
            gamma_cost=values.get("gamma_cost", 0.3),
            vrpp=values.get("vrpp", False),
            profit_aware_operators=values.get("profit_aware_operators", True),
            seed=values.get("seed", 42),
        )

        solver = DistancePSOSolver(
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
