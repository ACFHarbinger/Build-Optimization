from abc import abstractmethod
from collections import deque
from typing import Any, Deque, Optional

import numpy as np

from logic.src.policies.helpers.reinforcement_learning.agents.base import RLAgent


class ContextualBanditAgent(RLAgent):
    """
    Base class for Contextual Multi-Armed Bandit (CMAB) agents.

    Unlike traditional bandits, contextual bandits use external features (context)
    to inform action selection. This base class extends the standard RLAgent
    with bookkeeping for CMAB-specific metrics and histories.
    """

    def __init__(self, n_arms: int, feature_dim: int, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the contextual bandit agent.

        Args:
            n_arms: Number of available actions.
            feature_dim: Dimension of the context feature vector.
            seed: Optional seed for the random number generator.
            history_size: Size of the reward tracking buffer.
        """
        self.n_arms = n_arms
        self.d = feature_dim
        self.trials = 0
        self.history_size = history_size
        self.rng = np.random.default_rng(seed)

        # Performance tracking
        self.rewards: Deque[float] = deque(maxlen=history_size)
        self.actions: Deque[int] = deque(maxlen=history_size)

    # Optional exploration rate decay (if applicable).
    def decay_epsilon(self):
        pass

    @abstractmethod
    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        pass

    # Return the current weights/parameters of the agent.
    def get_weights(self) -> Any:
        return None

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def reset(self) -> None:
        self.trials = 0
        self.rewards.clear()
        self.actions.clear()
