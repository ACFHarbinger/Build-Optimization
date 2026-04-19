"""
RL-AHVPL Policy Adapter.

Adapts the Reinforcement Learning Augmented Hybrid Volleyball Premier League
(RL-AHVPL) logic to the agnostic policy interface.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.other import (
    BanditConfig,
    ContextFeatureExtractorConfig,
    EvolutionaryCMABConfig,
    FeatureExtractorConfig,
    LinUCBConfig,
    RewardShapingConfig,
    RLConfig,
    TDLearningConfig,
)
from logic.src.configs.policies.rl_ahvpl import RLAHVPLConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.base.factory import RouteConstructorRegistry
from logic.src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params import (
    RLAHVPLParams,
)
from logic.src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl import (
    RLAHVPLSolver,
)
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams
from logic.src.policies.route_construction.meta_heuristics.reactive_tabu_search.params import RTSParams


@GlobalRegistry.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.CONSTRUCTION,
    PolicyTag.PROFIT_AWARE,
)
@RouteConstructorRegistry.register("rl_ahvpl")
class RLAHVPLPolicy(BaseRoutingPolicy):
    """
    RL-AHVPL policy class.

    Visits pre-selected 'mandatory' bins using the Reinforcement Learning
    augmented HVPL metaheuristic combining ACO, ALNS, HGS, and CMAB.
    """

    def __init__(self, config: Optional[Union[RLAHVPLConfig, Dict[str, Any]]] = None):
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        return RLAHVPLConfig

    def _get_config_key(self) -> str:
        return "rl_ahvpl"

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
        Execute the Reinforcement Learning Augmented Hybrid Volleyball Premier League
        (RL-AHVPL) solver logic.

        RL-AHVPL is a highly sophisticated meta-hybrid that orchestrates multiple
        routing algorithms (ACO, ALNS, HGS, RTS) within a competitive/collaborative
        Volleyball Premier League framework. It uses an evolutionary CMAB or
        LinUCB agent to dynamically assign specific solvers or operator
        sequences to "teams" based on their historical performance on the
        current problem topology.

        The system incorporates elite coaching (intensification), substitutions
        (diversification), and various local search refinements to achieve
        state-of-the-art results for profit-aware VRP.

        Args:
            sub_dist_matrix (np.ndarray): Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes (Dict[int, float]): Mapping of local node indices to their
                current bin inventory levels.
            capacity (float): Maximum vehicle collection capacity.
            revenue (float): Revenue obtained per kilogram of waste collected.
            cost_unit (float): Monetary cost incurred per kilometer traveled.
            values (Dict[str, Any]): Merged configuration dictionary containing
                RL configurations and nested parameters for ACO, ALNS, HGS, and RTS.
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
            local_search=values.get("aco_local_search", False),
            local_search_iterations=values.get("aco_local_search_iterations", 0),
            elitist_weight=values.get("aco_elitist_weight", 1.0),
            time_limit=values.get("aco_time_limit", 60.0),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        alns_params = ALNSParams(
            max_iterations=values.get("alns_iterations", 500),
            start_temp=values.get("alns_start_temp", 100.0),
            cooling_rate=values.get("alns_cooling_rate", 0.95),
            reaction_factor=values.get("alns_reaction_factor", 0.1),
            min_removal=values.get("alns_min_removal", 1),
            max_removal_pct=values.get("alns_max_removal_pct", 0.2),
            time_limit=values.get("alns_time_limit", 60.0),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        hgs_params = HGSParams(
            time_limit=values.get("hgs_time_limit", 60.0),
            mu=values.get("hgs_mu", int(values.get("hgs_population_size", 50))),
            nb_elite=values.get("hgs_nb_elite", int(values.get("hgs_elite_size", 5))),
            mutation_rate=values.get("hgs_mutation_rate", 0.2),
            crossover_rate=values.get("hgs_crossover_rate", 0.7),
            n_offspring=values.get("hgs_n_offspring", int(values.get("hgs_n_generations", 100))),
            n_iterations_no_improvement=values.get(
                "hgs_n_iterations_no_improvement", int(values.get("hgs_no_improvement_threshold", 20))
            ),
            nb_granular=values.get("hgs_nb_granular", int(values.get("hgs_neighbor_list_size", 10))),
            local_search_iterations=values.get("hgs_local_search_iterations", 500),
            max_vehicles=values.get("hgs_max_vehicles", 0),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        rts_params = RTSParams(
            initial_tenure=values.get("rts_initial_tenure", 7),
            min_tenure=values.get("rts_min_tenure", 3),
            max_tenure=values.get("rts_max_tenure", 20),
            tenure_increase=values.get("rts_tenure_increase", 1.5),
            tenure_decrease=values.get("rts_tenure_decrease", 0.9),
            max_iterations=values.get("rts_max_iterations", 500),
            n_removal=values.get("rts_n_removal", 2),
            n_llh=values.get("rts_n_llh", 5),
            time_limit=values.get("rts_time_limit", 60.0),
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

        # Handle nested rl_config if present (from new Hydra structure)
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
                contextual=LinUCBConfig(**{k: v for k, v in rl_vals.get("contextual", {}).items() if k != "_target_"}),
                evolution_cmab=EvolutionaryCMABConfig(
                    **{k: v for k, v in rl_vals.get("evolution_cmab", {}).items() if k != "_target_"}
                ),
                reward=RewardShapingConfig(**{k: v for k, v in rl_vals.get("reward", {}).items() if k != "_target_"}),
                features=FeatureExtractorConfig(
                    **{k: v for k, v in rl_vals.get("features", {}).items() if k != "_target_"}
                ),
                context_features=ContextFeatureExtractorConfig(
                    **{k: v for k, v in rl_vals.get("context_features", {}).items() if k != "_target_"}
                ),
                params=rl_vals.get("params", {}),
            )
        else:
            # Build from flat values (legacy/fallback)
            rl_config = RLConfig(
                agent_type=values.get("bandit_algorithm", "linucb"),
                bandit=BanditConfig(
                    algorithm=values.get("bandit_algorithm", "linucb"),
                    seed=seed,
                ),
                td_learning=TDLearningConfig(
                    alpha=values.get("qlearning_alpha", 0.1),
                    gamma=values.get("qlearning_gamma", 0.9),
                    epsilon=values.get("qlearning_epsilon", 0.1),
                    epsilon_decay=values.get("qlearning_epsilon_decay", 0.99),
                    epsilon_decay_step=values.get("qlearning_epsilon_decay_step", 10),
                    epsilon_min=values.get("qlearning_epsilon_min", 0.05),
                    history_size=values.get("qlearning_history_size", 10),
                ),
                sarsa=TDLearningConfig(
                    alpha=values.get("sarsa_alpha", 0.1),
                    gamma=values.get("sarsa_gamma", 0.95),
                    epsilon=values.get("sarsa_epsilon", 0.15),
                    epsilon_decay=values.get("sarsa_epsilon_decay", 0.995),
                    epsilon_decay_step=values.get("sarsa_epsilon_decay_step", 50),
                    epsilon_min=values.get("sarsa_epsilon_min", 0.05),
                    history_size=values.get("sarsa_diversity_size", 10),
                ),
                contextual=LinUCBConfig(
                    alpha=values.get("cfe_alpha", 0.1),
                    feature_dim=values.get("cfe_feature_dim", 8),
                    lambda_prior=values.get("cfe_lambda_prior", 1.0),
                    noise_variance=values.get("cfe_noise_variance", 0.1),
                    history_size=values.get("reward_history_size", 50),
                ),
                evolution_cmab=EvolutionaryCMABConfig(
                    quality_weight=values.get("bandit_quality_weight", 0.5),
                    improvement_weight=values.get("bandit_improvement_weight", 1.0),
                    diversity_weight=values.get("bandit_diversity_weight", 0.2),
                    novelty_weight=values.get("bandit_novelty_weight", 1.0),
                    reward_threshold=values.get("bandit_reward_threshold", 1e-6),
                    default_reward=values.get("bandit_default_reward", 5.0),
                ),
                context_features=ContextFeatureExtractorConfig(
                    alpha=values.get("cfe_alpha", 0.1),
                    feature_dim=values.get("cfe_feature_dim", 8),
                    selection_threshold=values.get("cfe_operator_selection_threshold", 1e-9),
                    lambda_prior=values.get("cfe_lambda_prior", 1.0),
                    noise_variance=values.get("cfe_noise_variance", 0.1),
                    epsilon=values.get("cfe_epsilon", 0.15),
                    epsilon_decay=values.get("cfe_epsilon_decay", 0.995),
                    epsilon_decay_step=values.get("cfe_epsilon_decay_step", 20),
                    epsilon_min=values.get("cfe_epsilon_min", 0.05),
                ),
                features=FeatureExtractorConfig(
                    diversity_history_size=values.get("cfe_diversity_history_size", 10),
                    improvement_history_size=values.get("cfe_improvement_history_size", 10),
                ),
                reward=RewardShapingConfig(
                    improvement_threshold=values.get("cfe_improvement_threshold", 1e-6),
                ),
                params={
                    "bandit_max_iterations": values.get("bandit_max_iterations", 1000),
                    "cfe_operator_reward_size": values.get("cfe_operator_reward_size", 50),
                    "qlearning_rewards_size": values.get("qlearning_rewards_size", 20),
                    "qlearning_improvement_thresholds": values.get("qlearning_improvement_thresholds", (1e-4, -1e-4)),
                    "sarsa_scores_size": values.get("sarsa_scores_size", 50),
                    "sarsa_qtable_size_rate": values.get("sarsa_qtable_size_rate", 0.5),
                    "sarsa_improvement_thresholds": values.get("sarsa_improvement_thresholds", (-1e-6, 1e-6)),
                    "sarsa_operator_progress_thresholds": values.get(
                        "sarsa_operator_progress_thresholds", (0.33, 0.67)
                    ),
                    "sarsa_operator_stagnation_thresholds": values.get(
                        "sarsa_operator_stagnation_thresholds", (10, 30)
                    ),
                    "sarsa_operator_diversity_thresholds": values.get(
                        "sarsa_operator_diversity_thresholds", (0.3, 0.7)
                    ),
                },
            )

        params = RLAHVPLParams(
            n_teams=values.get("n_teams", 10),
            sub_rate=values.get("sub_rate", 0.2),
            time_limit=values.get("time_limit", 60.0),
            elite_coaching_max_iterations=values.get("alns_elite_iterations", 500),
            not_coached_max_iterations=values.get("alns_not_coached_iterations", 100),
            coaching_acceptance_threshold=values.get("coaching_acceptance_threshold", 1e-6),
            gls_probability=values.get("gls_probability", 0.5),
            seed=seed,
            hgs_params=hgs_params,
            aco_params=aco_params,
            alns_params=alns_params,
            rts_params=rts_params,
            rl_config=rl_config,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            tabu_no_repeat_threshold=values.get("tabu_no_repeat_threshold", 2),
            gls_penalty_lambda=values.get("gls_penalty_lambda", 1.0),
            gls_penalty_alpha=values.get("gls_penalty_alpha", 0.5),
            gls_penalty_step=values.get("gls_penalty_step", 10),
        )

        solver = RLAHVPLSolver(
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
