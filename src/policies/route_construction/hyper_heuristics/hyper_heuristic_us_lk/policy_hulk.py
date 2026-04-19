"""
Policy adapter for HULK hyper-heuristic.

Provides the interface between the simulator and the HULK solver.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.hulk import HULKConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .hulk import HULKSolver
from .params import HULKParams


@GlobalRegistry.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.ORCHESTRATOR,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("hulk")
class HULKPolicy(BaseRoutingPolicy):
    """HULK hyper-heuristic policy class."""

    def __init__(self, config: Optional[Union[HULKConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return HULKConfig

    def _get_config_key(self) -> str:
        return "hulk"

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
        Execute the Hyper-Heuristic Unstringing and Linked-K-opt (HULK) solver logic.

        HULK is a high-performance hyper-heuristic that adaptively selects between
        diverse unstringing (destroy) and stringing (reconstruct) operators. It
        leverages a multi-armed bandit (epsilon-greedy) approach with reinforcement
        learning to prioritize heuristics that contribute to objective improvements
        or search space exploration.

        The solver incorporates restarts, simulated annealing-style acceptance,
        and Linked-K-opt local search refinement to achieve near-optimal routing
        in VRPP instances.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                hyperparameters like `epsilon`, `weight_learning_rate`, etc.
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
        params = HULKParams(
            seed=values.get("seed", 42),
            max_iterations=int(values.get("max_iterations", 1000)),
            time_limit=float(values.get("time_limit", 300.0)),
            restarts=int(values.get("restarts", 3)),
            restart_threshold=int(values.get("restart_threshold", 100)),
            vrpp=bool(values.get("vrpp", True)),
            profit_aware_operators=bool(values.get("profit_aware_operators", False)),
            epsilon=float(values.get("epsilon", 0.3)),
            epsilon_decay=float(values.get("epsilon_decay", 0.995)),
            min_epsilon=float(values.get("min_epsilon", 0.05)),
            memory_size=int(values.get("memory_size", 50)),
            accept_worse_prob=float(values.get("accept_worse_prob", 0.1)),
            acceptance_decay=float(values.get("acceptance_decay", 0.99)),
            start_temp=float(values.get("start_temp", 100.0)),
            cooling_rate=float(values.get("cooling_rate", 0.99)),
            min_temp=float(values.get("min_temp", 0.01)),
            min_destroy_size=int(values.get("min_destroy_size", 2)),
            max_destroy_pct=float(values.get("max_destroy_pct", 0.3)),
            local_search_iterations=int(values.get("local_search_iterations", 10)),
            local_search_operators=values.get("local_search_operators", ["2-opt", "3-opt", "swap", "relocate"]),
            unstring_operators=values.get("unstring_operators", ["type_i", "type_ii", "type_iii", "type_iv"]),
            string_operators=values.get("string_operators", ["type_i", "type_ii", "type_iii", "type_iv"]),
            score_alpha=float(values.get("score_alpha", 10.0)),
            score_beta=float(values.get("score_beta", 5.0)),
            score_gamma=float(values.get("score_gamma", -1.0)),
            score_delta=float(values.get("score_delta", 20.0)),
            weight_learning_rate=float(values.get("weight_learning_rate", 0.1)),
            weight_decay=float(values.get("weight_decay", 0.95)),
        )

        solver = HULKSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        return solver.solve()
