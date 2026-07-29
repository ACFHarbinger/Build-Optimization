from typing import Any, Dict, List, Optional

import numpy as np

from .base import ContextualBanditAgent


class ContextualThompsonSamplingAgent(ContextualBanditAgent):
    """
    Contextual Thompson Sampling agent with linear Gaussian model.

    Bayesian approach to contextual bandits. Maintains a posterior distribution
    over the parameters (theta) for each arm.
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        lambda_prior: float = 1.0,
        noise_variance: float = 0.1,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Contextual Thompson Sampling agent.

        Args:
            n_arms: Number of available actions.
            feature_dim: Dimension of context feature vector (d).
            lambda_prior: Regularization parameter (initial precision).
            noise_variance: Estimated variance of the reward noise (sigma^2).
            seed: Optional random seed.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, feature_dim, seed, history_size)
        self.lambda_prior = lambda_prior
        self.v2 = noise_variance  # sigma^2

        # Precision matrices (B) and weighted reward vectors (f)
        # B_a is (d x d), f_a is (d x 1)
        self.B = [np.identity(self.d) * lambda_prior for _ in range(n_arms)]
        self.f = [np.zeros((self.d, 1)) for _ in range(n_arms)]

        # Cache for B_inv
        self.B_inv = [np.identity(self.d) / lambda_prior for _ in range(n_arms)]
        self._cache_valid = [True] * n_arms

    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select an arm using Thompson sampling from the linear posterior.

        Args:
            context: Current context feature vector [d].
            rng: Random number generator.

        Returns:
            The selected arm index.
        """
        x = context.reshape(-1, 1)
        samples = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            if not self._cache_valid[a]:
                self.B_inv[a] = np.linalg.inv(self.B[a])
                self._cache_valid[a] = True

            mu_a = self.B_inv[a] @ self.f[a]
            # Sample theta_a ~ N(mu_a, v^2 * B_inv_a)
            # Use Cholesky if needed, but since we are per-arm and d is small,
            # multivariate_normal is fine.
            theta_a = self.rng.multivariate_normal(mu_a.flatten(), self.v2 * self.B_inv[a])
            samples[a] = theta_a.T @ x.flatten()

        return int(np.argmax(samples))

    def update(
        self, context: np.ndarray, action: int, reward: float, next_context: Any = None, done: bool = False
    ) -> None:
        """
        Update the posterior distribution for the selected arm index.

        Args:
            context: Context vector associated with the choice.
            action: Selected arm index.
            reward: Observed reward.
            next_context: Unused.
            done: Unused.
        """
        x = context.reshape(-1, 1)
        self.B[action] += x @ x.T
        self.f[action] += reward * x
        self._cache_valid[action] = False

        self.trials += 1
        self.rewards.append(reward)
        self.actions.append(action)

    def reset(self) -> None:
        """Reset the posterior to the initial prior for all arms."""
        super().reset()
        self.B = [np.identity(self.d) * self.lambda_prior for _ in range(self.n_arms)]
        self.f = [np.zeros((self.d, 1)) for _ in range(self.n_arms)]
        self.B_inv = [np.identity(self.d) / self.lambda_prior for _ in range(self.n_arms)]
        self._cache_valid = [True] * self.n_arms

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic info for the TS agent.

        Returns:
            Dictionary containing trials, prior/noise parameters, and recent rewards.
        """
        return {
            "trials": self.trials,
            "lambda_prior": self.lambda_prior,
            "noise_variance": self.v2,
            "avg_reward": np.mean(self.rewards) if self.rewards else 0.0,
        }

    def get_weights(self) -> List[np.ndarray]:
        """Return the current posterior means for each arm."""
        weights = []
        for a in range(self.n_arms):
            if not self._cache_valid[a]:
                self.B_inv[a] = np.linalg.inv(self.B[a])
                self._cache_valid[a] = True
            weights.append(self.B_inv[a] @ self.f[a])
        return weights
