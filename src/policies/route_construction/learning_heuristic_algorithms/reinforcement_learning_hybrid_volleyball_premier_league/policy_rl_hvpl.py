"""
RL-HVPL Policy Adapter.

Adapts the Reinforcement Learning Hybrid Volleyball Premier League logic
to the agnostic policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.other import RLConfig
from logic.src.configs.policies.rl_hvpl import RLHVPLConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_hybrid_volleyball_premier_league.params import (
    RLHVPLParams,
)
from logic.src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl import (
    RLHVPLSolver,
)
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams


@GlobalRegistry.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("rl_hvpl")
class RLHVPLPolicy(BaseRoutingPolicy):
    """
    RL-HVPL policy class.

    Visits pre-selected 'mandatory' bins using the population-based RL-HVPL metaheuristic.
    """

    def __init__(self, config: Optional[Union[RLHVPLConfig, Dict[str, Any]]] = None):
        """Initialize RL-HVPL policy with optional config.

        Args:
            config: RLHVPLConfig dataclass, raw dict from YAML, or None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RLHVPLConfig

    def _get_config_key(self) -> str:
        """Return config key for RL-HVPL."""
        return "rl_hvpl"

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
        Execute the Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL)
        solver logic.

        RL-HVPL is a population-based metaheuristic inspired by the management of
        a volleyball league. It maintains a population of solutions ("teams")
        that evolve through "matches" (operator applications from ACO or ALNS).
        A reinforcement learning agent manages the transformation of teams,
        evaluating their relative strengths and applying coaching or
        substitutions to the worst-performing solution to drive global
        convergence towards higher profits.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                HVPL parameters, nested ACO parameters, and ALNS settings.
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
        seed = values.get("seed", 42)
        vrpp = values.get("vrpp", True)
        profit_aware_operators = values.get("profit_aware_operators", False)

        # Extract sub-params for ACO and ALNS
        aco_params = KSACOParams(
            n_ants=values.get("aco_n_ants", 10),
            k_sparse=values.get("aco_k_sparse", 10),
            alpha=values.get("aco_alpha", 1.0),
            beta=values.get("aco_beta", 2.0),
            rho=values.get("aco_rho", 0.1),
            q0=values.get("aco_q0", 0.9),
            tau_0=values.get("aco_tau_0"),
            tau_min=values.get("aco_tau_min", 0.001),
            tau_max=values.get("aco_tau_max", 10.0),
            max_iterations=values.get("aco_iterations", 1),
            time_limit=values.get("aco_time_limit", 60.0),
            local_search=values.get("aco_local_search", True),
            local_search_iterations=values.get("aco_local_search_iterations", 50),
            elitist_weight=values.get("aco_elitist_weight", 1.0),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        alns_params = ALNSParams(
            max_iterations=values.get("alns_iterations", 200),
            start_temp=values.get("alns_start_temp", 100.0),
            cooling_rate=values.get("alns_cooling_rate", 0.97),
            reaction_factor=values.get("alns_reaction_factor", 0.5),
            min_removal=values.get("alns_min_removal", 1),
            max_removal_pct=values.get("alns_max_removal_pct", 0.3),
            time_limit=values.get("alns_time_limit", 60.0),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        # RLConfig extraction (assuming it comes nested if mapped cleanly, but
        # handle flattened kwargs or a sub-dict correctly)
        rl_config = values.get("rl_config")
        if rl_config is None:
            rl_config = RLConfig()
        elif isinstance(rl_config, dict):
            # Try to build RLConfig from dict if passed organically
            rl_config = RLConfig(**rl_config)

        params = RLHVPLParams(
            n_teams=values.get("n_teams", 10),
            max_iterations=values.get("max_iterations", 100),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
            aco_params=aco_params,
            alns_params=alns_params,
            rl_config=rl_config,
            pheromone_update_strategy=values.get("pheromone_update_strategy", "profit"),
            profit_weight=values.get("profit_weight", 1.0),
            elite_coaching_iterations=values.get("elite_coaching_iterations", 300),
            regular_coaching_iterations=values.get("regular_coaching_iterations", 100),
            elite_size=values.get("elite_size", 3),
            seed=seed,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
        )

        solver = RLHVPLSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, solver_cost = solver.solve()
        return routes, profit, solver_cost
