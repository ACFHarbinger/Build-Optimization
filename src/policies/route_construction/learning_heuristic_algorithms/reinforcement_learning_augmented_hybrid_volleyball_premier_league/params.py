# Standard library
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

# Local imports
from logic.src.configs.policies.other import RLConfig
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.params import HGSParams
from logic.src.policies.route_construction.meta_heuristics.reactive_tabu_search.params import RTSParams


@dataclass
class RLAHVPLParams:
    """
    Parameters for the Reinforcement Learning Augmented Hybrid Volleyball Premier League metaheuristic.

    Combines VPL population dynamics, ACO initialization, ALNS local search, and HGS diversity
    management / crossover, all using Reinforcement Learning (RL) for adaptive operator selection.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    elite_coaching_max_iterations: int = 500
    not_coached_max_iterations: int = 100
    coaching_acceptance_threshold: float = 1e-6
    sub_rate: float = 0.2
    seed: Optional[int] = None
    vrpp: bool = False
    profit_aware_operators: bool = False

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # ACO parameters
    aco_params: KSACOParams = field(default_factory=KSACOParams)

    # ALNS parameters
    alns_params: ALNSParams = field(default_factory=ALNSParams)

    # HGS parameters
    hgs_params: HGSParams = field(default_factory=HGSParams)

    # Tabu parameters
    tabu_no_repeat_threshold: int = 2
    rts_params: RTSParams = field(default_factory=RTSParams)

    def __post_init__(self):
        """Sync flags across sub-parameters."""
        if self.aco_params:
            self.aco_params.vrpp = self.vrpp
            self.aco_params.profit_aware_operators = self.profit_aware_operators
        if self.alns_params:
            self.alns_params.vrpp = self.vrpp
            self.alns_params.profit_aware_operators = self.profit_aware_operators
        if self.hgs_params:
            self.hgs_params.vrpp = self.vrpp
            self.hgs_params.profit_aware_operators = self.profit_aware_operators
        if self.rts_params:
            self.rts_params.vrpp = self.vrpp
            self.rts_params.profit_aware_operators = self.profit_aware_operators

    # GLS parameters
    gls_penalty_lambda: float = 1.0
    gls_penalty_alpha: float = 0.5
    gls_penalty_step: int = 10
    gls_probability: float = 0.5

    # --- Legacy / Compatibility Properties ---
    # These properties allow existing code to access RL parameters without modification,
    # while transitioning to the new centralized RLConfig structure.

    def _get_val(self, category: str, key: str, default: Any = None) -> Any:
        # Resolve the category (e.g. td_learning, sarsa)
        if isinstance(self.rl_config, dict):
            cat_obj = self.rl_config.get(category, {})
        else:
            cat_obj = getattr(self.rl_config, category, None)

        # If no category found but we asked for sarsa, fallback to td_learning
        if not cat_obj and category == "sarsa":
            return self._get_val("td_learning", key, default)

        if isinstance(cat_obj, dict):
            return cat_obj.get(key, default)
        return getattr(cat_obj, key, default)

    def _get_param(self, key: str, default: Any = None) -> Any:
        if isinstance(self.rl_config, dict):
            return self.rl_config.get("params", {}).get(key, default)
        return self.rl_config.params.get(key, default)

    @property
    def bandit_algorithm(self) -> str:
        return self._get_val("bandit", "algorithm", "ucb1")

    @property
    def bandit_max_iterations(self) -> int:
        return self._get_param("bandit_max_iterations", 1000)

    @property
    def bandit_quality_weight(self) -> float:
        return self._get_val("evolution_cmab", "quality_weight", 0.5)

    @property
    def bandit_improvement_weight(self) -> float:
        return self._get_val("evolution_cmab", "improvement_weight", 1.0)

    @property
    def bandit_diversity_weight(self) -> float:
        return self._get_val("evolution_cmab", "diversity_weight", 0.2)

    @property
    def bandit_novelty_weight(self) -> float:
        return self._get_val("evolution_cmab", "novelty_weight", 1.0)

    @property
    def bandit_reward_threshold(self) -> float:
        return self._get_val("evolution_cmab", "reward_threshold", 1e-6)

    @property
    def bandit_default_reward(self) -> float:
        return self._get_val("evolution_cmab", "default_reward", 5.0)

    @property
    def cfe_alpha(self) -> float:
        return self._get_val("context_features", "alpha", 0.1)

    @property
    def cfe_feature_dim(self) -> int:
        return self._get_val("context_features", "feature_dim", 8)

    @property
    def cfe_operator_selection_threshold(self) -> float:
        return self._get_val("context_features", "selection_threshold", 1e-9)

    @property
    def cfe_lambda_prior(self) -> float:
        return self._get_val("context_features", "lambda_prior", 1.0)

    @property
    def cfe_noise_variance(self) -> float:
        return self._get_val("context_features", "noise_variance", 0.1)

    @property
    def cfe_epsilon(self) -> float:
        return self._get_val("context_features", "epsilon", 0.15)

    @property
    def cfe_epsilon_decay(self) -> float:
        return self._get_val("context_features", "epsilon_decay", 0.995)

    @property
    def cfe_epsilon_decay_step(self) -> int:
        return self._get_val("context_features", "epsilon_decay_step", 20)

    @property
    def cfe_epsilon_min(self) -> float:
        return self._get_val("context_features", "epsilon_min", 0.05)

    @property
    def cfe_diversity_history_size(self) -> int:
        return self._get_val("features", "diversity_history_size", 10)

    @property
    def cfe_improvement_history_size(self) -> int:
        return self._get_val("features", "improvement_history_size", 10)

    @property
    def cfe_operator_reward_size(self) -> int:
        return self._get_param("cfe_operator_reward_size", 50)

    @property
    def cfe_improvement_threshold(self) -> float:
        return float(self._get_val("reward", "improvement_threshold", 1e-6))

    @property
    def qlearning_alpha(self) -> float:
        return self._get_val("td_learning", "alpha", 0.1)

    @property
    def qlearning_gamma(self) -> float:
        return self._get_val("td_learning", "gamma", 0.95)

    @property
    def qlearning_epsilon(self) -> float:
        return self._get_val("td_learning", "epsilon", 0.1)

    @property
    def qlearning_epsilon_decay(self) -> float:
        return self._get_val("td_learning", "epsilon_decay", 0.995)

    @property
    def qlearning_epsilon_decay_step(self) -> int:
        return self._get_val("td_learning", "epsilon_decay_step", 20)

    @property
    def qlearning_epsilon_min(self) -> float:
        return self._get_val("td_learning", "epsilon_min", 0.05)

    @property
    def qlearning_history_size(self) -> int:
        return self._get_val("td_learning", "history_size", 100)

    @property
    def qlearning_rewards_size(self) -> int:
        return self._get_param("qlearning_rewards_size", 100)

    @property
    def qlearning_improvement_thresholds(self) -> Tuple[float, float]:
        val = self._get_param("qlearning_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def sarsa_alpha(self) -> float:
        return self._get_val("sarsa", "alpha", 0.1)

    @property
    def sarsa_gamma(self) -> float:
        return self._get_val("sarsa", "gamma", 0.95)

    @property
    def sarsa_epsilon(self) -> float:
        return self._get_val("sarsa", "epsilon", 0.1)

    @property
    def sarsa_epsilon_decay(self) -> float:
        return self._get_val("sarsa", "epsilon_decay", 0.995)

    @property
    def sarsa_epsilon_decay_step(self) -> int:
        return self._get_val("sarsa", "epsilon_decay_step", 20)

    @property
    def sarsa_epsilon_min(self) -> float:
        return self._get_val("sarsa", "epsilon_min", 0.05)

    @property
    def sarsa_diversity_size(self) -> int:
        return self._get_param("sarsa_diversity_size", 50)

    @property
    def sarsa_scores_size(self) -> int:
        return self._get_param("sarsa_scores_size", 100)

    @property
    def sarsa_qtable_size_rate(self) -> float:
        return self._get_param("sarsa_qtable_size_rate", 0.5)

    @property
    def sarsa_improvement_thresholds(self) -> Tuple[float, float]:
        val = self._get_param("sarsa_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def sarsa_operator_progress_thresholds(self) -> Tuple[float, float]:
        val = self._get_param("sarsa_operator_progress_thresholds", (0.3, 0.7))
        return float(val[0]), float(val[1])

    @property
    def sarsa_operator_stagnation_thresholds(self) -> Tuple[int, int]:
        val = self._get_param("sarsa_operator_stagnation_thresholds", (10, 50))
        return int(val[0]), int(val[1])

    @property
    def sarsa_operator_diversity_thresholds(self) -> Tuple[float, float]:
        val = self._get_param("sarsa_operator_diversity_thresholds", (0.2, 0.5))
        return float(val[0]), float(val[1])
