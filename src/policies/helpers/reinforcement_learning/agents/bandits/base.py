import pickle
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from logic.src.policies.helpers.reinforcement_learning.agents.base import RLAgent


class BanditAgent(RLAgent, ABC):
    """
    Base class for Multi-Armed Bandit (MAB) agents.

    Maintains basic statistics for each arm (action) including selection counts
    and estimated average rewards. Provides core mechanisms for state persistence
    and diagnostic reporting.

    Attributes:
        n_arms: The number of available arms (actions).
        counts: NumPy array of selection counts for each arm.
        values: NumPy array of current reward estimates for each arm.
        rng: Random number generator for deterministic stability.
        reward_history: Dictionary mapping arm indices to deques of recent rewards.
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the bandit agent.

        Args:
            n_arms: Number of available actions.
            seed: Optional seed for the random number generator.
            history_size: Maximum size of the reward tracking buffer for each arm.
        """
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.rng = np.random.default_rng(seed)

        # Performance tracking
        self.reward_history: Dict[int, Deque[float]] = {i: deque(maxlen=history_size) for i in range(n_arms)}

    def decay_epsilon(self) -> None:
        """
        Optional epsilon decay mechanism.

        Implemented by classes that utilize randomized exploration.
        """
        pass

    def reset(self) -> None:
        """Reset the agent's internal state."""
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)

    def save(self, path: str) -> None:
        """Save the agent's state to a file using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """Load the agent's state from a file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic statistics about the agent.

        Returns:
            Dictionary containing selection counts, values, and the best arm.
        """
        return {
            "counts": self.counts.tolist(),
            "values": self.values.tolist(),
            "best_arm": int(np.argmax(self.values)) if self.n_arms > 0 else None,
            "avg_history_rewards": {i: float(np.mean(h)) if h else 0.0 for i, h in self.reward_history.items()},
        }

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Standard incremental update for stationary environments.

        Formula: Q(a) = Q(a) + (1/n) * [r - Q(a)]

        Args:
            state: Initial state (unused).
            action: The selected arm index.
            reward: Observed reward.
            next_state: Resulting state (unused).
            done: Termination flag (unused).
        """
        # Track reward history
        self.reward_history[action].append(reward)

        # Incremental update
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n

    @abstractmethod
    def select_action(self, state: Any, rng: np.random.Generator) -> int:
        """
        Select an arm to pull based on the internal policy.

        This method must be implemented by subclasses (e.g., EpsilonGreedy, UCB).

        Args:
            state: Contextual state (for contextual bandits).
            rng: Random number generator for deterministic stability.

        Returns:
            The index of the selected arm (0-based).
        """
        pass
