"""
Hyperparameters for Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL).

This configuration bridges the basic HVPL (ACO + ALNS in population framework)
and the advanced RL-AHVPL (with HGS, CMAB, GLS), providing RL-enhanced operators
without the full genetic evolution complexity.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from logic.src.configs.policies.other import RLConfig
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params import ALNSParams
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams


@dataclass
class RLHVPLParams:
    """
    Parameters for the Reinforcement Learning Hybrid Volleyball Premier League.

    Combines:
    - ACO with Q-Learning for intelligent construction
    - ALNS with SARSA for adaptive destroy/repair operator selection
    - Population-based framework (VPL) for global search
    - NO genetic operators (simpler than RL-AHVPL)
    """

    # ===== General Parameters =====
    n_teams: int = 10  # Population size
    max_iterations: int = 100  # Number of league seasons
    sub_rate: float = 0.2  # Fraction of teams replaced per iteration
    time_limit: float = 60.0  # Overall time limit in seconds
    seed: Optional[int] = None
    vrpp: bool = False
    profit_aware_operators: bool = False

    # ===== RL Configuration (Centralized) =====
    rl_config: RLConfig = field(default_factory=RLConfig)

    def __post_init__(self):
        """Sync flags across sub-parameters."""
        if self.aco_params:
            self.aco_params.vrpp = self.vrpp
            self.aco_params.profit_aware_operators = self.profit_aware_operators
        if self.alns_params:
            self.alns_params.vrpp = self.vrpp
            self.alns_params.profit_aware_operators = self.profit_aware_operators

    # ===== ACO Parameters (with Q-Learning) =====
    aco_params: KSACOParams = field(
        default_factory=lambda: KSACOParams(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            q0=0.9,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Single construction per call
            time_limit=10.0,
            local_search=True,
            local_search_iterations=50,
            elitist_weight=1.0,
        )
    )

    # ===== ALNS Parameters (with SARSA) =====
    alns_params: ALNSParams = field(
        default_factory=lambda: ALNSParams(
            max_iterations=200,
            start_temp=100.0,
            cooling_rate=0.97,
            reaction_factor=0.5,
            min_removal=1,
            max_removal_pct=0.3,
            time_limit=20.0,
        )
    )

    # ===== Pheromone Update Strategy =====
    pheromone_update_strategy: str = "profit"  # "profit" or "cost"
    profit_weight: float = 1.0  # Weight for profit-based pheromone updates

    # ===== Coaching Parameters =====
    elite_coaching_iterations: int = 300  # Intensive coaching for top teams
    regular_coaching_iterations: int = 100  # Light coaching for others
    elite_size: int = 3  # Number of teams receiving elite coaching

    # --- Legacy / Compatibility Properties ---
    # These properties allow existing code to access RL parameters without modification,
    # maintaining consistency with RL-AHVPL interface.
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
    def qlearning_alpha(self) -> float:
        return float(self._get_val("td_learning", "alpha", 0.1))

    @property
    def qlearning_gamma(self) -> float:
        return float(self._get_val("td_learning", "gamma", 0.95))

    @property
    def qlearning_epsilon(self) -> float:
        return float(self._get_val("td_learning", "epsilon", 0.1))

    @property
    def qlearning_epsilon_decay(self) -> float:
        return float(self._get_val("td_learning", "epsilon_decay", 0.995))

    @property
    def qlearning_epsilon_decay_step(self) -> int:
        return self._get_val("td_learning", "epsilon_decay_step", 20)

    @property
    def qlearning_epsilon_min(self) -> float:
        return float(self._get_val("td_learning", "epsilon_min", 0.05))

    @property
    def qlearning_improvement_thresholds(self) -> Tuple[float, float]:
        val = self._get_param("qlearning_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def qlearning_history_size(self) -> int:
        return self._get_param("qlearning_history_size", 50)

    @property
    def sarsa_alpha(self) -> float:
        return float(self._get_val("sarsa", "alpha", 0.1))

    @property
    def sarsa_gamma(self) -> float:
        return float(self._get_val("sarsa", "gamma", 0.95))

    @property
    def sarsa_epsilon(self) -> float:
        return float(self._get_val("sarsa", "epsilon", 0.1))

    @property
    def sarsa_epsilon_decay(self) -> float:
        return float(self._get_val("sarsa", "epsilon_decay", 0.995))

    @property
    def sarsa_epsilon_decay_step(self) -> int:
        return self._get_val("sarsa", "epsilon_decay_step", 20)

    @property
    def sarsa_epsilon_min(self) -> float:
        return float(self._get_val("sarsa", "epsilon_min", 0.05))

    @property
    def sarsa_diversity_size(self) -> int:
        return self._get_param("sarsa_diversity_size", 50)

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
