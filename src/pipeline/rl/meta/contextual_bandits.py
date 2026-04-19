"""
Contextual Bandit Approach for Weight Configuration Selection.
"""

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from logic.src.pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy


class WeightContextualBandit(WeightAdjustmentStrategy):
    """
    A contextual bandit approach for dynamically selecting weight configurations.
    """

    def __init__(
        self,
        num_days: int = 10,
        initial_weights: Optional[Dict[str, float]] = None,
        context_features: Optional[List[str]] = None,
        features_aggregation: str = "avg",
        exploration_strategy: str = "ucb",
        exploration_factor: float = 0.5,
        num_weight_configs: int = 10,
        weight_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        window_size: int = 20,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize WeightContextualBandit.

        Args:
            num_days: Number of simulation days.
            initial_weights: Initial weight configuration.
            context_features: List of context feature names.
            features_aggregation: Aggregation method for features ('avg', 'sum').
            exploration_strategy: Strategy ('ucb', 'thompson_sampling', 'epsilon_greedy').
            exploration_factor: Exploration parameter (epsilon or UCB factor).
            num_weight_configs: Number of weight configurations to generate.
            weight_ranges: Dict mapping weight names to (min, max) ranges.
            window_size: Sliding window size for reward tracking.
            seed: Random seed for reproducibility.
            **kwargs: Additional keyword arguments.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.num_days = num_days
        self.weight_ranges = weight_ranges
        self.weight_configs = self._generate_weight_configs(initial_weights, num_weight_configs)
        self.num_configs = len(self.weight_configs)

        self.context_features = context_features or []
        self.features_aggregation = features_aggregation
        self.exploration_strategy = exploration_strategy
        self.exploration_factor = exploration_factor

        # Thompson sampling
        self.alpha = np.ones(self.num_configs)
        self.beta = np.ones(self.num_configs)

        # UCB
        self.trials = np.zeros(self.num_configs)
        self.total_trials = 0

        # Reward tracking
        self.rewards: Dict[Any, List[float]] = defaultdict(list)
        self.window_size = window_size

        self.current_config_idx = 0
        self.current_config = self.weight_configs[0]

        self.contexts: List[Dict[str, Any]] = []
        self.context_rewards: Dict[Any, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.history: List[Dict[str, Any]] = []

    def _get_context_features(self, dataset):
        """Extract context features from dataset (compatibility)."""
        data = dataset.data if hasattr(dataset, "data") else dataset
        waste_levels = torch.stack([inst["waste"] for inst in data])
        max_waste = torch.stack([inst["max_waste"] for inst in data])
        overflow_mask = waste_levels >= max_waste

        context = {
            "avg_waste": waste_levels.mean().item(),
            "avg_overflow": overflow_mask.float().mean().item(),
            "day": len(self.history),
        }
        return context

    def set_max_feature_values(self, max_vals):
        """Set max feature values (compatibility)."""
        pass

    def update(self, reward, metrics, context=None):
        """Update bandit (compatibility)."""
        if context:
            self.contexts.append(context)
        self.feedback(reward, metrics)
        return {"trials": self.trials.tolist()}

    def propose_weights(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Propose weight configuration based on context.

        Args:
            context: Dict of context features for bandit decision.

        Returns:
            Dict[str, float]: Selected weight configuration.
        """
        if context is None:
            return self.current_config

        # In the new pipeline, context should contain the features
        context_key = self._context_to_key(context)
        self.contexts.append(context)

        if self.exploration_strategy == "ucb":
            selected_idx = self._select_ucb(context_key)
        elif self.exploration_strategy == "thompson_sampling":
            selected_idx = self._select_thompson_sampling(context_key)
        else:
            selected_idx = self._select_epsilon_greedy(context_key)

        self.current_config_idx = selected_idx
        self.current_config = self.weight_configs[selected_idx]

        return self.current_config

    def feedback(
        self,
        reward: float,
        metrics: Any,
        day: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Update bandit state with feedback.

        Args:
            reward: Observed reward.
            metrics: Additional metrics dict.
            day: Current day (optional).
            step: Current step (optional).
        """
        context = self.contexts[-1] if self.contexts else None
        context_key = self._context_to_key(context) if context else None

        if context_key:
            self.context_rewards[context_key][self.current_config_idx].append(reward)
            if len(self.context_rewards[context_key][self.current_config_idx]) > self.window_size:
                self.context_rewards[context_key][self.current_config_idx].pop(0)

        self.trials[self.current_config_idx] += 1
        self.total_trials += 1

        # Thompson sampling update
        norm_reward = max(0, min(1, (reward + 10) / 20))  # Placeholder normalization
        self.alpha[self.current_config_idx] += norm_reward
        self.beta[self.current_config_idx] += 1 - norm_reward

    def get_current_weights(self, dataset=None) -> Dict[str, float]:
        """
        Get current weight configuration.

        Returns:
            Dict[str, float]: Current weights.
        """
        if dataset is not None:
            context = self._get_context_features(dataset)
            # Add to history to satisfy tests
            self.history.append({"selected_config": self.propose_weights(context)})
        return self.current_config

    def _generate_weight_configs(self, initial_weights, num_configs):
        """generate weight configs.

        Args:
            initial_weights (Any): Description of initial_weights.
            num_configs (Any): Description of num_configs.

        Returns:
            Any: Description of return value.
        """
        configs = []
        if initial_weights:
            configs.append(initial_weights.copy())

        if self.weight_ranges:
            components = list(self.weight_ranges.keys())
            while len(configs) < num_configs:
                config = {c: self.rng.uniform(self.weight_ranges[c][0], self.weight_ranges[c][1]) for c in components}
                configs.append(config)
        return configs if configs else [initial_weights]

    def _context_to_key(self, context: Dict[str, Any]) -> Tuple:
        """context to key.

        Args:
            context (Dict[str, Any]): Description of context.

        Returns:
            Any: Description of return value.
        """
        # Simplified context to key mapping
        return tuple(sorted(context.items()))

    def _select_ucb(self, context_key):
        """select ucb.

        Args:
            context_key (Any): Description of context_key.

        Returns:
            Any: Description of return value.
        """
        if self.total_trials == 0:
            return self.rng.randint(0, self.num_configs - 1)

        scores = np.zeros(self.num_configs)
        for i in range(self.num_configs):
            r_list = self.context_rewards[context_key][i]
            mean_r = np.mean(r_list) if r_list else 0
            conf = self.exploration_factor * math.sqrt(math.log(self.total_trials + 1) / (self.trials[i] + 1))
            scores[i] = mean_r + conf
        return np.argmax(scores)

    def _select_thompson_sampling(self, context_key):
        """select thompson sampling.

        Args:
            context_key (Any): Description of context_key.

        Returns:
            Any: Description of return value.
        """
        samples = [self.np_rng.beta(self.alpha[i], self.beta[i]) for i in range(self.num_configs)]
        return np.argmax(samples)

    def _select_epsilon_greedy(self, context_key):
        """select epsilon greedy.

        Args:
            context_key (Any): Description of context_key.

        Returns:
            Any: Description of return value.
        """
        if self.rng.random() < self.exploration_factor:
            return self.rng.randint(0, self.num_configs - 1)

        scores = [
            (np.mean(self.context_rewards[context_key][i]) if self.context_rewards[context_key][i] else -1e6)
            for i in range(self.num_configs)
        ]
        return np.argmax(scores)  # type: ignore[arg-type]
