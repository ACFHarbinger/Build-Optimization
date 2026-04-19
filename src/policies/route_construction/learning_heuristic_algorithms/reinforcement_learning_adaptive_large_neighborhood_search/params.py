# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

# Local imports
from logic.src.configs.policies.other import RLConfig

if TYPE_CHECKING:
    from logic.src.configs.policies import RLALNSConfig


@dataclass
class RLALNSParams:
    """
    Configuration parameters for the RL-ALNS solver.

    Attributes:
        time_limit: Maximum runtime in seconds.
        max_iterations: Maximum number of ALNS iterations.
        start_temp: Initial temperature for simulated annealing.
        cooling_rate: Temperature decay factor per iteration.
        min_removal: Minimum number of nodes to remove.
        max_removal_pct: Maximum percentage of nodes to remove.
        rl_config: Centralized Reinforcement Learning configuration.
    """

    # ALNS base parameters
    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    min_removal: int = 1
    max_removal_pct: float = 0.3
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # --- Legacy / Compatibility Properties ---

    @property
    def rl_algorithm(self) -> str:
        """Name of the RL algorithm being used."""
        # For ALNS, we might use both bandits or TD agents
        if self.rl_config.agent_type == "td_learning":
            return self.rl_config.td_learning.algorithm
        return self.rl_config.bandit.algorithm

    @property
    def alpha(self) -> float:
        return self.rl_config.td_learning.alpha

    @property
    def gamma(self) -> float:
        return self.rl_config.td_learning.gamma

    @property
    def epsilon(self) -> float:
        return self.rl_config.td_learning.epsilon

    @property
    def ucb_c(self) -> float:
        return self.rl_config.bandit.c

    @property
    def reward_new_global_best(self) -> float:
        return self.rl_config.reward.best_reward

    @property
    def reward_improved_current(self) -> float:
        return self.rl_config.reward.local_reward

    @property
    def reward_accepted_worse(self) -> float:
        return self.rl_config.reward.accepted_reward

    @property
    def reward_rejected(self) -> float:
        return self.rl_config.reward.rejected_reward

    @property
    def adaptive_rewards(self) -> bool:
        return self.rl_config.reward.adaptive_rewards

    @property
    def normalize_rewards(self) -> bool:
        return self.rl_config.reward.normalize_rewards

    @property
    def epsilon_decay(self) -> float:
        if self.rl_config.agent_type == "td_learning":
            return self.rl_config.td_learning.epsilon_decay
        return self.rl_config.bandit.epsilon_decay

    @property
    def epsilon_min(self) -> float:
        if self.rl_config.agent_type == "td_learning":
            return self.rl_config.td_learning.epsilon_min
        return self.rl_config.bandit.epsilon_min

    @property
    def ucb_gamma(self) -> float:
        return self.rl_config.bandit.gamma

    @property
    def ucb_window_size(self) -> int:
        return self.rl_config.bandit.window_size

    @property
    def ts_alpha_prior(self) -> float:
        return self.rl_config.bandit.alpha_prior

    @property
    def ts_beta_prior(self) -> float:
        return self.rl_config.bandit.beta_prior

    @property
    def exp3_gamma(self) -> float:
        return self.rl_config.bandit.gamma

    @property
    def progress_thresholds(self) -> list[float]:
        return self.rl_config.features.progress_thresholds

    @property
    def stagnation_thresholds(self) -> list[int]:
        return self.rl_config.features.stagnation_thresholds

    @property
    def diversity_thresholds(self) -> list[float]:
        return self.rl_config.features.diversity_thresholds

    @classmethod
    def from_config(cls, config: RLALNSConfig) -> RLALNSParams:
        """
        Create RLALNSParams from a RLALNSConfig dataclass.
        Note: This remains for backward compatibility with Hydra loaders.
        """
        return cls(
            time_limit=config.time_limit,
            max_iterations=config.max_iterations,
            start_temp=config.start_temp,
            cooling_rate=config.cooling_rate,
            min_removal=config.min_removal,
            max_removal_pct=config.max_removal_pct,
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            rl_config=config.rl_config,
        )
