"""
Hyper-ACO Policy Adapter.

Adapts the Hyper-Heuristic ACO solver to the common policy interface.
"""

import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import HyperHeuristicACOConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
    HyperHeuristicACO,
)
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import (
    OPERATOR_NAMES,
)
from logic.src.policies.route_construction.hyper_heuristics.ant_colony_optimization_hyper_heuristic.params import (
    HyperACOParams,
)


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.SWARM_INTELLIGENCE,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("aco_hh")
class HyperACOPolicy(BaseRoutingPolicy):
    """
    Hyper-Heuristic ACO policy class.

    Uses ACO to construct sequences of local search operators.
    """

    def __init__(self, config: Optional[Union[HyperHeuristicACOConfig, Dict[str, Any]]] = None):
        """Initialize Hyper-ACO policy with optional config.

        Args:
            config: HyperHeuristicACOConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HyperHeuristicACOConfig

    def _get_config_key(self) -> str:
        """Return config key for Hyper-ACO."""
        return "hyper_aco"

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
        Execute the Hyper-Heuristic ACO (Hyper-ACO) solver logic.

        Hyper-ACO uses an Ant Colony Optimization framework where pheromone
        trails are deposited on sequences of local search and ruin-and-recreate
        operators rather than on problem edges. This allows the system to
        "learn" which combinations of heuristics are most effective for the
        current problem topology.

        The best sequence discovered by the ants is applied to a greedy initial
        solution to produce the final optimized plan.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                hyperparameters like `n_ants`, `alpha`, `beta`, `rho`, etc.
            mandatory_nodes (List[int]): Local indices of bins that MUST be
                collected in this period.
            **kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        # Parse parameters
        params = HyperACOParams(
            n_ants=values.get("n_ants", 10),
            alpha=values.get("alpha", 1.0),
            beta=values.get("beta", 2.0),
            rho=values.get("rho", 0.1),
            tau_0=values.get("tau_0", 1.0),
            tau_min=values.get("tau_min", 0.01),
            tau_max=values.get("tau_max", 10.0),
            max_iterations=values.get("max_iterations", 50),
            time_limit=values.get("time_limit", 30.0),
            q0=values.get("q0", 0.9),
            lambda_factor=values.get("lambda_factor", 1.0001),
            operators=values.get("operators", OPERATOR_NAMES.copy()),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # Determine initial routes
        initial_routes = build_greedy_routes(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            mandatory_nodes=mandatory_nodes,
            rng=random.Random(params.seed) if params.seed is not None else random.Random(),
        )

        solver = HyperHeuristicACO(
            sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, params, initial_routes, mandatory_nodes
        )
        return solver.solve()
