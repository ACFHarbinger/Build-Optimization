from typing import Any, Dict, Optional

import numpy as np

from .base import BanditAgent


class EXP3Agent(BanditAgent):
    """
    EXP3 algorithm for adversarial bandits.

    Exponential-weight algorithm for Exploration and Exploitation. Designed to
    handle adversarial environments where the rewards are not drawn from a
    fixed distribution.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.1,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the EXP3 agent.

        Args:
            n_arms: Number of available actions.
            gamma: Exploration parameter (controls the mixture between weights and uniform dist).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.probs = np.ones(n_arms) / n_arms

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the EXP3 policy.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        rng = rng or self.rng

        # Calculate probabilities based on current weights
        total_w = np.sum(self.weights)
        # Prob = (1-gamma) * (weight / total_weight) + (gamma / n_arms)
        self.probs = (1 - self.gamma) * (self.weights / (total_w + 1e-12)) + (self.gamma / self.n_arms)

        # Sample according to probabilities
        return int(rng.choice(self.n_arms, p=self.probs))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update the weights of the selected arm.

        Args:
            state: Initial state.
            action: Selected arm index.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Standard tracking
        self.reward_history[action].append(reward)
        self.counts[action] += 1

        # Scale reward to [0, 1] for stable exponential updates
        rew = np.clip((reward + 10) / 20, 0, 1)

        # Importance-weighted reward estimate: e_r = r / prob(action)
        est_reward = rew / (self.probs[action] + 1e-12)

        # Update weight: w = w * exp(gamma * est_reward / n_arms)
        self.weights[action] *= np.exp(self.gamma * est_reward / self.n_arms)

    def reset(self) -> None:
        """Reset weights and probabilities to uniform distribution."""
        super().reset()
        self.weights = np.ones(self.n_arms)
        self.probs = np.ones(self.n_arms) / self.n_arms

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with EXP3 specific weights and probabilities.

        Returns:
            Dictionary including basic stats, weights, and current probabilities.
        """
        stats = super().get_statistics()
        stats.update({"weights": self.weights.tolist(), "probs": self.probs.tolist()})
        return stats
