"""
Reinforcement Learning Great Deluge Hyper-Heuristic (RL-GD-HH) Policy Adapter.

This module provides the integration layer between the RLGDHHSolver and the
broader simulator infrastructure. It adapts the learning-based hyper-heuristic
engine to the agnostic BaseRoutingPolicy interface.

Reference:
    Ozcan, E., Misir, M., Ochoa, G., & Burke, E. K. (2010).
    "A Reinforcement Learning – Great-Deluge Hyper-heuristic for Examination Timetabling".
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.rl_gd_hh import RLGDHHConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry

from .params import RLGDHHParams
from .solver import RLGDHHSolver


@GlobalRegistry.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("rl_gd_hh")
class RLGDHHPolicy(BaseRoutingPolicy):
    """
    Adapter for the RL-GD-HH solver.

    This policy handles the conversion from abstract problem variables
    (distance matrices, revenue/cost units) into the specific parameters
    and state required by the hyper-heuristic engine.

    Design Pattern: Adapter
    ----------------------
    RLGDHHPolicy acts as a bridge, allowing the learning-based solver
    implementation to be used within the simulator's standardized routing
    framework.
    """

    def __init__(self, config: Optional[Union[RLGDHHConfig, Dict[str, Any]]] = None):
        """
        Initialize the RLGDHHPolicy adapter.

        Args:
            config: An optional configuration object or dictionary containing
                    the engine hyper-parameters.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Defines the Hydra-compliant configuration schema for this policy.

        Returns:
            RLGDHHConfig class.
        """
        return RLGDHHConfig

    def _get_config_key(self) -> str:
        """
        Returns the unique identifier for this policy in the registry.

        Returns:
            "rl_gd_hh"
        """
        return "rl_gd_hh"

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
        Execute the Reinforcement Learning - Great Deluge Hyper-Heuristic (RL-GD-HH)
        solver logic.

        RL-GD-HH employs an online learning mechanism to dynamically select
        low-level heuristics based on their past success. The Great Deluge
        component regulates solution acceptance using a global "water level"
        threshold that decreases over time, facilitating an effective balance
        between intensification and diversification without the need for
        parameter-heavy temperature cooling schedules.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                RL parameters and Great Deluge settings like `flood_margin`.
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
        # 1. Parameter Extraction (Mapping simulator values to RLGDHHParams)
        params = RLGDHHParams(
            max_iterations=int(values.get("max_iterations", 5000)),
            time_limit=float(values.get("time_limit", 60.0)),
            reward_improvement=float(values.get("reward_improvement", 1.0)),
            penalty_worsening=float(values.get("penalty_worsening", 1.0)),
            punishment_type=str(values.get("punishment_type", "RL1")),
            utility_upper_bound=float(values.get("utility_upper_bound", 40.0)),
            min_utility=float(values.get("min_utility", 0.0)),
            initial_utility=float(values.get("initial_utility", 30.0)),
            quality_lb=float(values.get("quality_lb", 0.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        # 2. Solver Initialization
        solver = RLGDHHSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        # 3. Learning-based Optimization (Ozcan et al. 2010 Algorithm)
        best_routes, best_reward, best_cost = solver.solve()

        # 4. Result Formatting
        total_cost = best_cost * cost_unit
        return best_routes, best_reward, total_cost
