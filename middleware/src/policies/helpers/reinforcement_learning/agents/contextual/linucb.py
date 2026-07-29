import contextlib
from typing import Any, Dict, List, Optional

import numpy as np

from .base import ContextualBanditAgent


class LinUCBAgent(ContextualBanditAgent):
    """
    Linear Upper Confidence Bound (LinUCB) agent.

    Maintains a linear model per arm for expected rewards given context.
    Algorithm: LinUCB with Disjoint Linear Models.

    References:
        Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010).
        A contextual-bandit approach to personalized news article recommendation.
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        alpha: float = 1.0,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the LinUCB agent.

        Args:
            n_arms: Number of available arms.
            feature_dim: Dimension of context feature vector (x).
            alpha: Exploration parameter (controls width of UCB bonus).
            seed: Optional random seed.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, feature_dim, seed, history_size)
        self.alpha = alpha

        # Per-arm parameters: A_a = M_a' * M_a + I_d, b_a = M_a' * r_a
        # A_a is (d x d), b_a is (d x 1)
        self.A = [np.identity(self.d) for _ in range(n_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(n_arms)]

        # Cache inverses for efficiency
        self.A_inv = [np.identity(self.d) for _ in range(n_arms)]
        self._cache_valid = [True] * n_arms

    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select an arm using the LinUCB policy.

        Args:
            context: Current context feature vector [d].
            rng: Random number generator for tie-breaking.

        Returns:
            The selected arm index.
        """
        x = context.reshape(-1, 1)  # (d x 1)
        p = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            if not self._cache_valid[a]:
                self.A_inv[a] = np.linalg.inv(self.A[a])
                self._cache_valid[a] = True

            theta_a = self.A_inv[a] @ self.b[a]  # (d x 1)
            mean = theta_a.T @ x
            uncertainty = self.alpha * np.sqrt(x.T @ self.A_inv[a] @ x)
            p[a] = mean + uncertainty

        # Break ties randomly
        max_p = np.max(p)
        best_arms = np.where(np.abs(p - max_p) < 1e-9)[0]
        return rng.choice(best_arms)

    def update(
        self, context: np.ndarray, action: int, reward: float, next_context: Any = None, done: bool = False
    ) -> None:
        """
        Update the linear model for the selected arm with observed reward.

        Args:
            context: Context vector associated with the choice.
            action: Index of the selected arm.
            reward: Observed reward.
            next_context: Unused in bandits.
            done: Unused in bandits.
        """
        x = context.reshape(-1, 1)
        self.A[action] += x @ x.T
        self.b[action] += reward * x
        self._cache_valid[action] = False

        self.trials += 1
        self.rewards.append(reward)
        self.actions.append(action)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic info for the LinUCB agent.

        Returns:
            Dictionary containing trial count, parameters, and recent performance.
        """
        stats = {
            "trials": self.trials,
            "alpha": self.alpha,
            "avg_reward": np.mean(self.rewards) if self.rewards else 0.0,
            "history_len": len(self.rewards),
        }
        return stats

    def get_weights(self) -> List[np.ndarray]:
        """
        Calculate current theta estimates (weights) for all arms.

        Returns:
            List of NumPy arrays, one per arm.
        """
        weights = []
        for a in range(self.n_arms):
            with contextlib.suppress(Exception):
                if not self._cache_valid[a]:
                    self.A_inv[a] = np.linalg.inv(self.A[a])
                    self._cache_valid[a] = True
            weights.append(self.A_inv[a] @ self.b[a])
        return weights
