import pickle
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Optional

import numpy as np

from .base import RLAgent


class TDAgent(RLAgent, ABC):
    """
    Base class for Temporal Difference (TD) learning agents.

    Supports tabular RL where the state-action space is represented as a Q-table.
    Uses a dictionary for the Q-table to efficiently support large or sparse
    state spaces.

    Attributes:
        n_states: Number of possible states (if known).
        n_actions: Number of available actions.
        alpha: Learning rate (step size).
        gamma: Discount factor for future rewards.
        epsilon: Exploration rate for epsilon-greedy selection.
        epsilon_decay: Multiplicative factor for epsilon reduction.
        epsilon_min: Lower bound for epsilon.
        q_table: Dictionary mapping states to NumPy arrays of Q-values.
        trial_counts: Dictionary mapping states to NumPy arrays of visitation counts.
        reward_history: Deque of observed global rewards for performance tracking.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        history_size: int = 100,
        seed: int = 42,
    ):
        """
        Initialize the TD Agent.

        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            alpha: Learning rate.
            gamma: Discount factor.
            epsilon: Exploration probability.
            epsilon_decay: Epsilon decay factor.
            epsilon_min: Minimum epsilon.
            history_size: Size of global reward tracking buffer.
            seed: Seed for random number generator.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seed = seed

        # Tabular storage
        self.q_table: Dict[Any, np.ndarray] = {}
        self.trial_counts: Dict[Any, np.ndarray] = {}

        # Meta-tracking
        self.reward_history: Deque[float] = deque(maxlen=history_size)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve diagnostic info for the TD agent.

        Returns:
            Dictionary containing table size, parameters, and exploration rate.
        """
        return {
            "q_table_entries": len(self.q_table),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "avg_reward": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            "total_trials": sum(np.sum(v) for v in self.trial_counts.values()) if self.trial_counts else 0,
        }

    def get_q_values(self, state: Any) -> np.ndarray:
        """
        Return Q-values for all actions in a given state.

        Initializes entries for new states on-the-fly.

        Args:
            state: The environment state identifier.

        Returns:
            NumPy array of q-values for the state.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            self.trial_counts[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def select_action(self, state: Any, rng: np.random.Generator) -> int:
        """
        Select an action using the epsilon-greedy policy.

        Args:
            state: Current environment state.
            rng: Random number generator.

        Returns:
            The selected action index.
        """
        # Exploration: Select random action with probability epsilon
        if rng.random() < self.epsilon:
            return int(rng.integers(0, self.n_actions))

        # Exploitation: Select action with highest Q-value
        q_values = self.get_q_values(state)
        max_q = np.max(q_values)
        # Handle ties by selecting randomly among the best actions
        best_actions = np.where(q_values == max_q)[0]

        return int(rng.choice(best_actions))

    def decay_epsilon(self) -> None:
        """Apply multiplicative decay to the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save the agent's state to a file using pickle."""
        state = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Load the agent's state from a file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.q_table = state["q_table"]
        self.epsilon = state["epsilon"]
        self.n_states = state.get("n_states", self.n_states)
        self.n_actions = state.get("n_actions", self.n_actions)

    def reset(self) -> None:
        """Reset the Q-table and exploration rate."""
        self.q_table = {}
        # We don't necessarily want to reset epsilon to initial value here,
        # but the interface allows it if needed. For now, just clear Q-table.

    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Perform a Temporal Difference update based on a transition.

        This method must be implemented by subclasses to define the specific
        TD update logic (e.g., Q-Learning, SARSA, Expected SARSA).

        Args:
            state: Initial state of the transition.
            action: Action taken in 'state'.
            reward: Scalar reward observed after taking 'action'.
            next_state: Resulting state after taking 'action'.
            done: Boolean flag indicating if the episode has terminated.
        """
        pass


class QLearningAgent(TDAgent):
    """
    Q-Learning Agent (Off-policy TD control).

    Directly learns the optimal (target) action-value function, independent
    of the policy followed by the agent.

    Formula: Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
    """

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Perform Q-Learning update based on transition.

        Args:
            state: Initial state.
            action: Action taken.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Global tracking
        self.reward_history.append(reward)
        self.get_q_values(state)  # ensure initialized
        self.trial_counts[state][action] += 1

        # Current estimation
        current_q = self.q_table[state][action]

        # Calculate target: Immediate reward + discounted max future reward
        if done:
            # If terminal state reached, no future return
            target = reward
        else:
            # Off-policy: use maximum possible reward in the next state
            next_max_q = np.max(self.get_q_values(next_state))
            target = reward + self.gamma * next_max_q

        # TD update using step size alpha
        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error


class SarsaAgent(TDAgent):
    """
    SARSA Agent (On-policy TD control).

    Learns the action-value function of the local execution policy.
    Name derived from the transition sequence: (S, A, R, S', A').

    Formula: Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize SARSA agent with next-action buffer."""
        super().__init__(*args, **kwargs)
        self.next_action: Optional[int] = None

    def update(
        self, state: Any, action: int, reward: float, next_state: Any, done: bool, next_action: Optional[int] = None
    ) -> None:
        """
        Perform SARSA update based on transition.

        Args:
            state: Initial state.
            action: Action taken.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
            next_action: The action selected to be taken in 'next_state'.
        """
        # Global tracking
        self.reward_history.append(reward)
        self.get_q_values(state)
        self.trial_counts[state][action] += 1

        # Current estimation
        current_q = self.q_table[state][action]

        # On-policy update needs the action taken in the next state
        if done:
            target = reward
        else:
            # Prefer the provided next_action, fallback to internal buffer or sample
            if next_action is not None:
                self.next_action = next_action

            if self.next_action is None:
                # Fallback: sample from epsilon-greedy policy for next state
                self.next_action = self.select_action(next_state, np.random.default_rng(self.seed))

            # Use specifically the next action's Q-value (on-policy)
            next_q = self.get_q_values(next_state)[self.next_action]
            target = reward + self.gamma * next_q

        # TD update
        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error

        # Reset next_action for the next iteration step
        self.next_action = None


class ExpectedSarsaAgent(TDAgent):
    """
    Expected SARSA Agent (On-policy TD control).

    Reduces variance by considering the expected value of the next state
    over all possible actions under the current policy, rather than a single sample.

    Formula: Q(s,a) = Q(s,a) + alpha * [r + gamma * E[Q(s',a')] - Q(s,a)]
    """

    def update(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """
        Perform Expected SARSA update.

        Args:
            state: Initial state.
            action: Action taken.
            reward: Observed reward.
            next_state: Resulting state.
            done: Termination flag.
        """
        # Tracking
        self.reward_history.append(reward)
        self.get_q_values(state)
        self.trial_counts[state][action] += 1

        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)

            # Calculate expectation under epsilon-greedy policy:
            # E[Q] = (1-eps) * Q(best) + eps * avg(Q(all))
            n_actions = self.n_actions
            best_q = np.max(next_q_values)
            n_best = np.sum(next_q_values == best_q)

            # Assign probabilities to each action under epsilon-greedy
            probs = np.full(n_actions, self.epsilon / n_actions)
            probs[next_q_values == best_q] += (1.0 - self.epsilon) / n_best

            # Weighted sum of Q-values
            expected_next_q = np.dot(probs, next_q_values)
            target = reward + self.gamma * expected_next_q

        # TD update
        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error
