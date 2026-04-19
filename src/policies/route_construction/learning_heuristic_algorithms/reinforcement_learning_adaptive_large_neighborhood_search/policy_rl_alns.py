from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.configs.policies.other import (
    BanditConfig,
    FeatureExtractorConfig,
    RewardShapingConfig,
    RLConfig,
    TDLearningConfig,
)
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.params import (
    RLALNSParams,
)
from logic.src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver import (
    RLALNSSolver,
)


@GlobalRegistry.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.META_HEURISTIC,
    PolicyTag.LARGE_NEIGHBORHOOD_SEARCH,
    PolicyTag.ORCHESTRATOR,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("rl_alns")
class RLALNSPolicy(BaseRoutingPolicy):
    """
    Adapter for the RL-ALNS solver.

    Bridges the centralized RLConfig structure with the RLALNSSolver.
    """

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
        Execute the Reinforcement Learning Adaptive Large Neighborhood Search (RL-ALNS)
        solver logic.

        RL-ALNS replaces the standard frequency-based weights in ALNS with an
        online reinforcement learning agent (Multi-Armed Bandit or Temporal
        Difference learner). The agent selects ruin-and-recreate operators based
        on their observed performance rewards (e.g., finding a new global best
        or improving the current solution).

        The search process is controlled by a simulated annealing acceptance
        criterion to balance intensification and diversification.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                RL configurations and ALNS parameters like `start_temp`, `cooling_rate`.
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
        # Handle nested rl_config if present (from new Hydra structure)
        seed = values.get("seed", 42)
        rl_vals = values.get("rl_config")
        if isinstance(rl_vals, dict):
            # Safe way to convert nested dict to RLConfig
            rl_config = RLConfig(
                agent_type=rl_vals.get("agent_type", "bandit"),
                bandit=BanditConfig(
                    **{k: v for k, v in rl_vals.get("bandit", {}).items() if k != "_target_"},
                    seed=seed,
                ),
                td_learning=TDLearningConfig(
                    **{k: v for k, v in rl_vals.get("td_learning", {}).items() if k != "_target_"}
                ),
                sarsa=TDLearningConfig(**{k: v for k, v in rl_vals.get("sarsa", {}).items() if k != "_target_"})
                if rl_vals.get("sarsa")
                else None,
                reward=RewardShapingConfig(**{k: v for k, v in rl_vals.get("reward", {}).items() if k != "_target_"}),
                features=FeatureExtractorConfig(
                    **{k: v for k, v in rl_vals.get("features", {}).items() if k != "_target_"}
                ),
                params=rl_vals.get("params", {}),
            )
        else:
            # Fallback for flat config structure (legacy)
            rl_config = RLConfig(
                agent_type=values.get("agent_type", "bandit"),
                bandit=BanditConfig(
                    algorithm=values.get("rl_algorithm", "ucb1"),
                    c=values.get("ucb_c", 2.0),
                    seed=seed,
                ),
                td_learning=TDLearningConfig(
                    alpha=values.get("alpha", 0.1),
                    gamma=values.get("gamma", 0.95),
                    epsilon=values.get("epsilon", 0.1),
                ),
                reward=RewardShapingConfig(
                    best_reward=values.get("reward_new_global_best", 10.0),
                    local_reward=values.get("reward_improved_current", 5.0),
                    accepted_reward=values.get("reward_accepted_worse", 1.0),
                    rejected_reward=values.get("reward_rejected", -1.0),
                ),
            )

        params = RLALNSParams(
            time_limit=values.get("time_limit", 60.0),
            max_iterations=values.get("max_iterations", 5000),
            start_temp=values.get("start_temp", 100.0),
            cooling_rate=values.get("cooling_rate", 0.995),
            min_removal=values.get("min_removal", 1),
            max_removal_pct=values.get("max_removal_pct", 0.3),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=seed,
            rl_config=rl_config,
        )

        solver = RLALNSSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        return solver.solve()
