from typing import Any, Dict, Optional

import numpy as np

from .base import BanditAgent


class ThompsonSamplingBandit(BanditAgent):
    """
    Thompson Sampling Bandit policy for Bernoulli environments.

    Uses Beta distribution priors for each arm's success probability.
    Selects actions by sampling from the posterior and playing the arm
    with the highest sampled success rate.
    """

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Thompson Sampling agent.

        Args:
            n_arms: Number of available actions.
            alpha_prior: Initial success count for Beta distribution.
            beta_prior: Initial failure count for Beta distribution.
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.alphas = np.full(n_arms, alpha_prior, dtype=float)
        self.betas = np.full(n_arms, beta_prior, dtype=float)

    def select_action(self, state: Any, rng: np.random.Generator) -> int:
        """
        Select an arm using Thompson sampling.

        Args:
            state: Contextual state (unused).
            rng: Random number generator.

        Returns:
            The selected arm index.
        """
        # Sample from the Beta distribution of each arm
        samples = [rng.beta(self.alphas[i], self.betas[i]) for i in range(self.n_arms)]

        # Select the arm with the highest sample
        return int(np.argmax(samples))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update the Beta posterior for the selected arm.

        Args:
            state: Initial state.
            action: The arm that was played.
            reward: Observed binary reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Standard bandit update for counts/values
        super().update(state, action, reward, next_state, done)

        # Update Beta parameters based on reward (assumes binary reward)
        # Success count (alpha) increases with reward, failure (beta) increases otherwise
        if reward > 0.5:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1

    def reset(self) -> None:
        """Reset internal parameters and success/failure counts."""
        super().reset()
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with Beta distribution parameters.

        Returns:
            Dictionary including basic stats and posterior alpha/beta values.
        """
        stats = super().get_statistics()
        stats.update({"alphas": self.alphas.tolist(), "betas": self.betas.tolist()})
        return stats
