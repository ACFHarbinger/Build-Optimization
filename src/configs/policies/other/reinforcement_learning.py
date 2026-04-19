"""
Reinforcement Learning Configuration module.

Defines unified configuration dataclasses for reinforcement learning agents,
reward shaping, and feature extraction components used across optimization policies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BanditConfig:
    """
    Configuration for Multi-Armed Bandit (MAB) agents.

    Attributes:
        algorithm: The bandit algorithm to use (ucb1, thompson, epsilon_greedy, etc.).
        epsilon: Initial exploration probability for epsilon-greedy.
        epsilon_decay: Factor to reduce epsilon over time.
        epsilon_min: Minimum allowable exploration probability.
        c: Exploration parameter for UCB algorithms.
        temperature: Temperature for Softmax sampling.
        alpha_prior: Initial success count for Thompson Sampling (Beta).
        beta_prior: Initial failure count for Thompson Sampling (Beta).
        gamma: Discount factor for Discounted/EXP3 agents.
        window_size: Observation window for SlidingWindowUCB.
        history_size: Size of reward tracking buffer for diagnostics.
        seed: Random seed for deterministic reproducibility.
    """

    algorithm: str = "ucb1"
    epsilon: float = 0.1
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01
    c: float = 2.0
    temperature: float = 1.0
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    gamma: float = 0.95
    window_size: int = 100
    history_size: int = 50
    seed: Optional[int] = None


@dataclass
class TDLearningConfig:
    """
    Configuration for Temporal Difference (TD) learning agents.

    Attributes:
        algorithm: The TD algorithm (q_learning, sarsa, expected_sarsa).
        alpha: Learning rate (step size).
        gamma: Discount factor for future rewards.
        epsilon: Exploration rate for epsilon-greedy selection.
        epsilon_decay: Multiplicative factor for epsilon reduction.
        epsilon_min: Lower bound for epsilon.
        history_size: Size of global reward tracking buffer.
        n_states: Number of discretized states in the tabular representation.
        n_actions: Number of possible actions (operators).
    """

    algorithm: str = "q_learning"
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    epsilon_decay_step: int = 20
    history_size: int = 100
    n_states: int = 27
    n_actions: int = 10


@dataclass
class LinUCBConfig:
    """
    Configuration for Contextual Multi-Armed Bandits (LinUCB).

    Attributes:
        alpha: Exploration parameter (controls the width of the confidence bound).
        feature_dim: Dimension of the context feature vector.
        lambda_prior: Regularization parameter for the ridge regression.
        noise_variance: Variance of the observation noise (optional).
        history_size: Size of reward history buffer.
    """

    alpha: float = 1.0
    feature_dim: int = 8
    lambda_prior: float = 1.0
    noise_variance: float = 0.1
    history_size: int = 50


@dataclass
class EvolutionaryCMABConfig:
    """
    Configuration for Contextual MABs used in evolutionary algorithms (e.g., crossover).

    Attributes:
        quality_weight: Weight for offspring quality in reward.
        improvement_weight: Weight for improvement over parents.
        diversity_weight: Weight for population diversity contribution.
        novelty_weight: Weight for genetic novelty.
        reward_threshold: Minimum improvement to consider a reward.
        default_reward: Baseline reward multiplier.
    """

    quality_weight: float = 0.5
    improvement_weight: float = 1.0
    diversity_weight: float = 0.2
    novelty_weight: float = 1.0
    reward_threshold: float = 1e-6
    default_reward: float = 5.0


@dataclass
class RewardShapingConfig:
    """
    Configuration for search outcome reward shaping.

    Attributes:
        best_reward: Reward for finding a new globally best solution.
        local_reward: Reward for local improvement over the current solution.
        accepted_reward: Base reward for an accepted step.
        rejected_reward: Penalty for a rejected move.
        stagnation_penalty: Penalty per iteration of stagnation.
        improvement_threshold: Minimum difference to consider as an improvement.
        rewards_size: Size of the rewards tracking buffer.
    """

    best_reward: float = 10.0
    local_reward: float = 5.0
    accepted_reward: float = 1.0
    rejected_reward: float = -1.0
    stagnation_penalty: float = -0.1
    adaptive_rewards: bool = False
    normalize_rewards: bool = False
    improvement_threshold: float = 1e-6
    rewards_size: int = 20


@dataclass
class FeatureExtractorConfig:
    """
    Configuration for state feature extraction and discretization.

    Attributes:
        progress_thresholds: Percentage points to split search phases (early/mid/late).
        stagnation_thresholds: Iteration counts to split stagnation levels (low/med/high).
        diversity_thresholds: Value ranges to split solution diversity levels.
        diversity_history_size: Buffer size for population diversity tracking.
        improvement_history_size: Buffer size for improvement velocity tracking.
    """

    progress_thresholds: List[float] = field(default_factory=lambda: [0.33, 0.67])
    stagnation_thresholds: List[int] = field(default_factory=lambda: [10, 30])
    diversity_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.7])
    diversity_history_size: int = 10
    improvement_history_size: int = 10


@dataclass
class ContextFeatureExtractorConfig:
    """
    Configuration for Context Feature Extractor (CFE).

    Attributes:
        alpha: Alpha parameter for feature extraction.
        feature_dim: Context dimension.
        selection_threshold: Minimum activation for operator selection.
        lambda_prior: Bayesian prior lambda.
        noise_variance: Noise variance for contextual sampling.
        epsilon: Initial exploration rate.
        epsilon_decay: Epsilon decay factor.
        epsilon_decay_step: Frequency of epsilon decay.
        epsilon_min: Minimum exploration rate.
    """

    alpha: float = 0.1
    feature_dim: int = 8
    selection_threshold: float = 1e-9
    lambda_prior: float = 1.0
    noise_variance: float = 0.1
    epsilon: float = 0.15
    epsilon_decay: float = 0.995
    epsilon_decay_step: int = 20
    epsilon_min: float = 0.05


@dataclass
class GPCMABConfig:
    """
    Configuration for Gaussian Process Combinatorial Multi-Armed Bandits.

    Attributes:
        beta: Exploration parameter for GP-UCB (standard deviations).
        length_scale: Length scale for the RBF kernel.
        signal_variance: Signal variance for the RBF kernel.
        noise_variance: Observation noise variance.
        max_history: Maximum number of points to keep in GP history.
        super_arm_size: Number of arms to select in a super-arm.
    """

    beta: float = 2.0
    length_scale: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 0.1
    max_history: int = 500
    super_arm_size: int = 1


@dataclass
class RLConfig:
    """
    Unified Reinforcement Learning configuration.

    Composes agent-specific parameters, reward shaping, and feature extraction
    into a single structured object.

    Attributes:
        agent_type: Type of RL agent ('bandit', 'td_learning', 'contextual', 'gp_cmab').
        bandit: Configuration for MAB agents.
        td_learning: Configuration for TD agents (e.g., Q-Learning).
        sarsa: Optional configuration for SARSA agents.
        contextual: Configuration for contextual bandits (LinUCB).
        gp_cmab: Configuration for GP-CMAB agents.
        evolution_cmab: Configuration for evolutionary MABs.
        reward: Configuration for reward shaping.
        features: Configuration for state feature extraction.
        context_features: Configuration for context feature extraction.
        params: Additional unstructured parameters for flexibility.
    """

    agent_type: str = "bandit"
    bandit: BanditConfig = field(default_factory=BanditConfig)
    td_learning: TDLearningConfig = field(default_factory=TDLearningConfig)
    sarsa: Optional[TDLearningConfig] = None
    contextual: LinUCBConfig = field(default_factory=LinUCBConfig)
    gp_cmab: GPCMABConfig = field(default_factory=GPCMABConfig)
    evolution_cmab: EvolutionaryCMABConfig = field(default_factory=EvolutionaryCMABConfig)
    reward: RewardShapingConfig = field(default_factory=RewardShapingConfig)
    features: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    context_features: ContextFeatureExtractorConfig = field(default_factory=ContextFeatureExtractorConfig)
    params: Dict[str, Any] = field(default_factory=dict)
