import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ContextualBanditAgent


class GPCMABAgent(ContextualBanditAgent):
    """
    Gaussian Process Combinatorial Multi-Armed Bandit (GP-CMAB) agent.

    Models the reward function as a Gaussian Process (GP) over the joint
    (context, action) space. Uses the GP-UCB acquisition function for
    arm selection, balancing exploration (high posterior variance) with
    exploitation (high posterior mean). Supports combinatorial "super-arm"
    selection where multiple base arms are chosen simultaneously.
    """

    def __init__(
        self,
        n_arms: int,
        feature_dim: int,
        beta: float = 2.0,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.1,
        max_history: int = 500,
        super_arm_size: int = 1,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the GP-CMAB agent.
        """
        super().__init__(n_arms, feature_dim, seed, history_size)
        self.beta = beta
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.max_history = max_history
        self.super_arm_size = min(super_arm_size, n_arms)

        # The GP input dimension is context + one-hot action encoding
        self.input_dim = feature_dim + n_arms

        # Observation history
        self.X_history: List[np.ndarray] = []
        self.y_history: List[float] = []

        # Cached kernel matrix inverse and alpha vector
        self.K_inv: Optional[np.ndarray] = None
        self.alpha_vec: Optional[np.ndarray] = None

    def _encode_input(self, context: np.ndarray, action: int) -> np.ndarray:
        one_hot = np.zeros(self.n_arms)
        one_hot[action] = 1.0
        return np.concatenate([context, one_hot])

    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        sq_dist = np.sum((x1 - x2) ** 2)
        return float(self.signal_variance * np.exp(-sq_dist / (2.0 * self.length_scale**2)))

    def _kernel_vector(self, x_star: np.ndarray) -> np.ndarray:
        if not self.X_history:
            return np.array([])
        return np.array([self._rbf_kernel(x_star, x_i) for x_i in self.X_history])

    def _predict(self, context: np.ndarray, action: int) -> Tuple[float, float]:
        x_star = self._encode_input(context, action)
        if self.K_inv is None or len(self.X_history) == 0:
            return 0.0, self.signal_variance

        k_star = self._kernel_vector(x_star)
        mu = float(k_star @ self.alpha_vec)
        k_star_star = self._rbf_kernel(x_star, x_star)
        v = self.K_inv @ k_star
        var = k_star_star - float(k_star @ v)
        var = max(var, 1e-10)
        return mu, var

    def _recompute_kernel_inverse(self) -> None:
        n = len(self.X_history)
        if n == 0:
            self.K_inv = None
            self.alpha_vec = None
            return

        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k_val = self._rbf_kernel(self.X_history[i], self.X_history[j])
                K[i, j] = k_val
                K[j, i] = k_val

        K += self.noise_variance * np.eye(n)
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))

        self.alpha_vec = self.K_inv @ np.array(self.y_history)

    def select_action(self, context: np.ndarray, rng: np.random.Generator) -> int:
        self.trials += 1
        ucb_scores = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            mu, var = self._predict(context, arm)
            ucb_scores[arm] = mu + np.sqrt(self.beta) * np.sqrt(var)

        if self.super_arm_size == 1:
            max_score = np.max(ucb_scores)
            best_arms = np.where(np.abs(ucb_scores - max_score) < 1e-9)[0]
            return int(rng.choice(best_arms))

        remaining_scores = ucb_scores.copy()
        for _ in range(self.super_arm_size):
            max_score = np.max(remaining_scores)
            best = np.where(np.abs(remaining_scores - max_score) < 1e-9)[0]
            chosen = int(rng.choice(best))
            remaining_scores[chosen] = -np.inf
            return chosen
        return int(rng.integers(0, self.n_arms))

    def select_super_arm(self, context: np.ndarray, rng: np.random.Generator) -> List[int]:
        self.trials += 1
        ucb_scores = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            mu, var = self._predict(context, arm)
            ucb_scores[arm] = mu + np.sqrt(self.beta) * np.sqrt(var)

        selected_arms = []
        remaining_scores = ucb_scores.copy()
        for _ in range(self.super_arm_size):
            max_score = np.max(remaining_scores)
            best = np.where(np.abs(remaining_scores - max_score) < 1e-9)[0]
            chosen = int(rng.choice(best))
            selected_arms.append(chosen)
            remaining_scores[chosen] = -np.inf
        return selected_arms

    def update(
        self, context: np.ndarray, action: int, reward: float, next_context: Any = None, done: bool = False
    ) -> None:
        x_new = self._encode_input(context, action)
        if len(self.X_history) >= self.max_history:
            self.X_history.pop(0)
            self.y_history.pop(0)
            self.K_inv = None
            self.alpha_vec = None

        self.X_history.append(x_new)
        self.y_history.append(reward)
        n = len(self.X_history)

        if n == 1:
            kappa = self._rbf_kernel(x_new, x_new) + self.noise_variance
            self.K_inv = np.array([[1.0 / kappa]])
            self.alpha_vec = np.array([reward / kappa])
        elif self.K_inv is not None and self.K_inv.shape[0] == n - 1:
            k = np.array([self._rbf_kernel(x_new, x_i) for x_i in self.X_history[:-1]])
            kappa = self._rbf_kernel(x_new, x_new) + self.noise_variance
            K_inv_k = self.K_inv @ k
            s = kappa - float(k @ K_inv_k)
            s = max(s, 1e-10)
            K_inv_new = np.zeros((n, n))
            K_inv_new[: n - 1, : n - 1] = self.K_inv + np.outer(K_inv_k, K_inv_k) / s
            K_inv_new[: n - 1, n - 1] = -K_inv_k / s
            K_inv_new[n - 1, : n - 1] = -K_inv_k / s
            K_inv_new[n - 1, n - 1] = 1.0 / s
            self.K_inv = K_inv_new
        else:
            self._recompute_kernel_inverse()

        self.alpha_vec = self.K_inv @ np.array(self.y_history)
        self.rewards.append(reward)
        self.actions.append(action)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "trials": self.trials,
            "n_arms": self.n_arms,
            "beta": self.beta,
            "n_observations": len(self.X_history),
            "avg_reward": np.mean(self.rewards) if self.rewards else 0.0,
        }

    def get_weights(self) -> Optional[np.ndarray]:
        return self.alpha_vec.copy() if self.alpha_vec is not None else None

    def save(self, path: str) -> None:
        state = {
            "n_arms": self.n_arms,
            "feature_dim": self.d,
            "beta": self.beta,
            "length_scale": self.length_scale,
            "X_history": self.X_history,
            "y_history": self.y_history,
            "K_inv": self.K_inv,
            "alpha_vec": self.alpha_vec,
            "trials": self.trials,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.X_history = state["X_history"]
        self.y_history = state["y_history"]
        self.K_inv = state["K_inv"]
        self.alpha_vec = state["alpha_vec"]
        self.trials = state.get("trials", 0)

    def reset(self) -> None:
        super().reset()
        self.X_history = []
        self.y_history = []
        self.K_inv = None
        self.alpha_vec = None
