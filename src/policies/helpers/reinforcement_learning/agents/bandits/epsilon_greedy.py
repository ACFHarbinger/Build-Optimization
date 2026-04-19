from typing import Any, Dict, Optional

import numpy as np

from .base import BanditAgent


class EpsilonGreedyBandit(BanditAgent):
    """
    Epsilon-Greedy Bandit policy.

    Balances exploration and exploitation by selecting a random arm with
    probability 'epsilon' and the best known arm with probability '1 - epsilon'.
    Supports dynamic epsilon decay to favor exploitation as search progresses.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Epsilon-Greedy agent.

        Args:
            n_arms: Number of available actions.
            epsilon: Initial exploration probability.
            epsilon_decay: Multiplicative factor for reducing epsilon.
            epsilon_min: Minimum allowable exploration probability.
            seed: Optional seed for the random number generator.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def decay_epsilon(self) -> None:
        """Apply multiplicative decay to the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def select_action(self, state: Any, rng: np.random.Generator) -> int:
        """
        Select an arm using the epsilon-greedy policy.

        Args:
            state: Contextual state (unused by non-contextual bandits).
            rng: Random number generator for local selection logic.

        Returns:
            The selected arm index.
        """
        # Exploration: Select a random arm with probability epsilon
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_arms))

        # Exploitation: Select the arm with the highest estimated value
        return int(np.argmax(self.values))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment basic statistics with current epsilon.

        Returns:
            Dictionary including basic stats and current exploration rate.
        """
        stats = super().get_statistics()
        stats["epsilon"] = self.epsilon
        return stats
