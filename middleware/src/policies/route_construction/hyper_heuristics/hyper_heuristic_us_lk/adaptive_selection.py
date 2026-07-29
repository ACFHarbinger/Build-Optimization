"""
Adaptive operator selection for HULK hyper-heuristic.

Implements credit assignment and selection mechanisms for choosing
unstringing/stringing and local search operators adaptively.
"""

import random
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np


class AdaptiveOperatorSelector:
    """
    Adaptive selection mechanism for HULK operators.

    Uses performance tracking and epsilon-greedy selection to balance
    exploration and exploitation of operators.
    """

    def __init__(
        self,
        operators: List[str],
        epsilon: float = 0.3,
        memory_size: int = 50,
        learning_rate: float = 0.1,
        weight_decay: float = 0.95,
        seed: Optional[int] = 42,
    ):
        """
        Initialize adaptive selector.

        Args:
            operators: List of operator names.
            epsilon: Exploration rate (0-1).
            memory_size: Window size for performance tracking.
            learning_rate: Rate of weight updates.
            weight_decay: Decay factor for historical weights.
            seed: Random seed.
        """
        self.operators = operators
        self.epsilon = epsilon
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = random.Random(seed) if seed is not None else random.Random()

        # Initialize tracking structures
        self.weights: Dict[str, float] = {op: 1.0 for op in operators}
        self.scores: Dict[str, Deque[float]] = {op: deque(maxlen=memory_size) for op in operators}
        self.times: Dict[str, Deque[float]] = {op: deque(maxlen=memory_size) for op in operators}
        self.applications: Dict[str, int] = {op: 0 for op in operators}
        self.improvements: Dict[str, int] = {op: 0 for op in operators}

    def select_operator(self) -> str:
        """
        Select an operator using epsilon-greedy strategy.

        Returns:
            Selected operator name.
        """
        if self.rng.random() < self.epsilon:
            # Exploration: random selection
            return self.rng.choice(self.operators)
        else:
            # Exploitation: select based on weights
            return self._select_by_weight()

    def _select_by_weight(self) -> str:
        """Select operator proportional to weights."""
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return self.rng.choice(self.operators)

        # Normalize weights to probabilities
        probs = [self.weights[op] / total_weight for op in self.operators]

        # Roulette wheel selection
        return self.rng.choices(self.operators, weights=probs, k=1)[0]

    def update(
        self,
        operator: str,
        improvement: float,
        elapsed_time: float,
        is_best: bool = False,
    ):
        """
        Update operator statistics after application.

        Args:
            operator: Name of applied operator.
            improvement: Improvement in objective (can be negative).
            elapsed_time: Time taken to apply operator.
            is_best: Whether this led to a new best solution.
        """
        if operator not in self.operators:
            return

        # Record application
        self.applications[operator] += 1

        # Calculate score
        score = self._calculate_score(improvement, is_best)

        # Update tracking
        self.scores[operator].append(score)
        self.times[operator].append(elapsed_time)

        if improvement > 0:
            self.improvements[operator] += 1

        # Update weight
        self._update_weight(operator)

    def _calculate_score(self, improvement: float, is_best: bool) -> float:
        """
        Calculate score (reward) for operator application.

        Paper levels (Müller & Bonilha 2022):
        - New best global: r1
        - Improved current: r2
        - Not improved: r3

        Returns:
            Score value.
        """
        if is_best:
            return 11.0  # r1
        elif improvement > 1e-6:
            return 5.0  # r2
        else:
            return 1.0  # r3

    def _update_weight(self, operator: str):
        """Update operator weight based on recent performance."""
        scores = list(self.scores[operator])
        if not scores:
            return

        # Calculate average recent score
        avg_score = np.mean(scores)

        # Apply decay to current weight
        self.weights[operator] *= self.weight_decay

        # Add learning component
        self.weights[operator] += self.learning_rate * float(avg_score)

        # Ensure positive weight
        self.weights[operator] = max(0.1, self.weights[operator])

    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.05):
        """Decay exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def get_statistics(self) -> Dict[str, Dict]:
        """
        Get performance statistics for all operators.

        Returns:
            Dictionary of operator statistics.
        """
        stats = {}
        for op in self.operators:
            avg_score = np.mean(list(self.scores[op])) if self.scores[op] else 0.0
            avg_time = np.mean(list(self.times[op])) if self.times[op] else 0.0
            success_rate = self.improvements[op] / self.applications[op] if self.applications[op] > 0 else 0.0

            stats[op] = {
                "weight": self.weights[op],
                "applications": self.applications[op],
                "improvements": self.improvements[op],
                "avg_score": avg_score,
                "avg_time": avg_time,
                "success_rate": success_rate,
            }

        return stats

    def reset_statistics(self):
        """Reset all operator statistics."""
        for op in self.operators:
            self.scores[op].clear()
            self.times[op].clear()
            self.applications[op] = 0
            self.improvements[op] = 0
            self.weights[op] = 1.0
