from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .base import BanditAgent


class UCBBandit(BanditAgent):
    """
    Upper Confidence Bound (UCB1) Bandit policy.

    The UCB1 algorithm implements 'optimism in the face of uncertainty' by
    adding a confidence bonus to the estimated value of each arm. This bonus
    decreases as the arm is selected more frequently.
    """

    def __init__(self, n_arms: int, c: float = 2.0, seed: Optional[int] = None, history_size: int = 50):
        """
        Initialize the UCB1 agent.

        Args:
            n_arms: Number of available actions.
            c: Exploration parameter (controls the width of the confidence interval).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.c = c

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the UCB1 policy.

        Rules:
            1. Select each arm at least once (bootstrap).
            2. For subsequent trials, maximize V_j + c * sqrt(ln(total_trials) / counts_j).

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        # Ensure every arm is explored at least once
        if 0 in self.counts:
            return int(np.where(self.counts == 0)[0][0])

        # Calculate UCB values for all arms
        total_counts = np.sum(self.counts)
        # Confidence bonus: bonus increases with total trials, decreases with arm-specific trials
        bonus = self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-12))
        ucb_values = self.values + bonus

        return int(np.argmax(ucb_values))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with exploration constant 'c'.

        Returns:
            Dictionary including basic stats and UCB parameter.
        """
        stats = super().get_statistics()
        stats["c"] = self.c
        return stats


class DiscountedUCBBandit(BanditAgent):
    """
    Discounted UCB Bandit policy.

    Useful for non-stationary environments where the optimal action may change
    over time. Uses a discount factor 'gamma' to slowly decay the contribution
    of past observations.
    """

    def __init__(
        self,
        n_arms: int,
        c: float = 2.0,
        gamma: float = 0.95,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Discounted UCB agent.

        Args:
            n_arms: Number of available actions.
            c: Exploration parameter.
            gamma: Discount factor (0 < gamma <= 1).
            seed: Optional seed for the RNG.
            history_size: Size of reward history buffer.
        """
        super().__init__(n_arms, seed, history_size)
        self.c = c
        self.gamma = gamma
        self.discounted_counts = np.zeros(n_arms)
        self.discounted_rewards = np.zeros(n_arms)

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the Discounted UCB policy.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        # Bootstrap: ensure every arm has at least some discounted representation
        if np.any(self.discounted_counts < 1.0):
            return int(np.argmin(self.discounted_counts))

        # Calculate UCB values using discounted statistics
        total_discounted = np.sum(self.discounted_counts)
        ucb_values = self.values + self.c * np.sqrt(np.log(total_discounted) / (self.discounted_counts + 1e-9))

        return int(np.argmax(ucb_values))

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Update discounted statistics for all arms.

        Args:
            state: Initial state.
            action: Selected arm.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Apply discount to all arm stats
        self.discounted_counts *= self.gamma
        self.discounted_rewards *= self.gamma

        # Increment stats for the selected arm
        self.discounted_counts[action] += 1
        self.discounted_rewards[action] += reward

        # Update current value estimate using discounted mean
        self.values[action] = self.discounted_rewards[action] / self.discounted_counts[action]

        # Use super().update if tracking of raw counts/history is desired
        # but avoid the incremental mean update which is inappropriate here
        self.reward_history[action].append(reward)
        self.counts[action] += 1


class SlidingWindowUCBBandit(BanditAgent):
    """
    Sliding Window UCB Bandit policy.

    Handles volatile environments by selecting arms based only on the most
    recent 'window_size' observations.
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 100,
        c: float = 2.0,
        seed: Optional[int] = None,
        history_size: int = 50,
    ):
        """
        Initialize the Sliding Window UCB agent.

        Args:
            n_arms: Number of available actions.
            window_size: Number of past observations to consider.
            c: Exploration parameter.
            seed: Optional seed for the RNG.
            history_size: Size of diagnostic reward history.
        """
        super().__init__(n_arms, seed, history_size)
        self.window_size = window_size
        self.c = c

        # Maintain window-specific reward deques
        self.windows: List[Deque[float]] = [deque(maxlen=window_size) for _ in range(n_arms)]

    def select_action(self, state: Any, rng: Optional[np.random.Generator] = None) -> int:
        """
        Select an arm using the Sliding Window UCB policy.

        Args:
            state: Contextual state (unused).
            rng: Optional RNG.

        Returns:
            The selected arm index.
        """
        counts = [len(w) for w in self.windows]

        # Ensure every arm has at least one observation in its current window
        if 0 in counts:
            return int(counts.index(0))

        # Calculate UCB values based only on the counts and values within the window
        total = sum(counts)
        ucb_values = [
            self.values[i] + self.c * np.sqrt(np.log(total) / len(self.windows[i])) for i in range(self.n_arms)
        ]

        return int(np.argmax(ucb_values))

    def reset(self) -> None:
        """Clear all windows and reset stats."""
        super().reset()
        self.windows = [deque(maxlen=self.window_size) for _ in range(self.n_arms)]

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Add observed reward to the window and update the mean estimate.

        Args:
            state: Initial state.
            action: Selected arm index.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Update sliding window
        self.windows[action].append(reward)

        # Update value using the mean of the current window
        self.values[action] = np.mean(self.windows[action])

        # Track raw statistics
        self.counts[action] += 1
        self.reward_history[action].append(reward)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Augment statistics with the current contents of the sliding windows.

        Returns:
            Dictionary including basic stats and the rewards in each window.
        """
        stats = super().get_statistics()
        stats.update({"windows": [list(w) for w in self.windows]})
        return stats
