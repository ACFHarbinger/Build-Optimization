"""
Multi-Objective Reinforcement Learning (MORL) Weight Optimizer.
"""

import copy
from collections import deque
from typing import Dict, List, Optional

import numpy as np
from logic.src.pipeline.rl.meta.multi_objective.pareto_front import ParetoFront
from logic.src.pipeline.rl.meta.multi_objective.pareto_solution import ParetoSolution
from logic.src.pipeline.rl.meta.weight_strategy import WeightAdjustmentStrategy


class MORLWeightOptimizer(WeightAdjustmentStrategy):
    """Pareto-based multi-objective weight optimizer."""

    def __init__(
        self,
        initial_weights: Dict[str, float],
        weight_names: Optional[List[str]] = None,
        objective_names: Optional[List[str]] = None,
        history_window: int = 20,
        exploration_factor: float = 0.2,
        adaptation_rate: float = 0.1,
        **kwargs,
    ):
        """
        Initialize MORLWeightOptimizer.

        Args:
            initial_weights: Initial weight configuration.
            weight_names: Names of weight components to optimize.
            objective_names: Names of objectives to track.
            history_window: Window size for performance history.
            exploration_factor: Probability of random exploration.
            adaptation_rate: Rate of weight perturbation.
            **kwargs: Additional keyword arguments.
        """
        if objective_names is None:
            objective_names = ["waste_efficiency", "overflow_rate"]
        if weight_names is None:
            weight_names = ["collection", "cost"]
        self.weight_names = weight_names
        self.objective_names = objective_names
        self.current_weights = copy.deepcopy(initial_weights)
        self.history_window = history_window
        self.performance_history: deque[Dict[str, float]] = deque(maxlen=history_window)
        self.exploration_factor = exploration_factor
        self.adaptation_rate = adaptation_rate
        self.pareto_front = ParetoFront()

    def propose_weights(self, context=None):
        """
        Propose weight configuration with optional exploration.

        Args:
            context: Optional context dict (unused).

        Returns:
            Dict[str, float]: Proposed weights.
        """
        if np.random.random() < self.exploration_factor:
            # Random perturbation
            for name in self.weight_names:
                self.current_weights[name] *= 1 + (np.random.random() - 0.5) * self.adaptation_rate
        return self.current_weights

    def _calculate_objectives(self, metrics):
        """Calculate objectives for the Pareto front (compatibility)."""
        den1 = metrics.get("tour_length", metrics.get("cost", 1))
        waste_eff = metrics.get("waste_collected", metrics.get("collection", 0)) / den1 if den1 != 0 else 0

        den2 = metrics.get("total_bins", 1)
        overflow_rate = metrics.get("num_overflows", metrics.get("overflows", 0)) / den2 if den2 != 0 else 0
        return {"waste_efficiency": waste_eff, "overflow_rate": overflow_rate}

    def update_performance_history(self, metrics, reward):
        """Update performance history (compatibility)."""
        self.feedback(reward, metrics)

    def feedback(self, reward, metrics, day=None, step=None):
        """
        Update optimizer with feedback.

        Args:
            reward: Observed reward.
            metrics: Metrics dict with objective values.
            day: Current day (optional).
            step: Current step (optional).
        """
        objectives = self._calculate_objectives(metrics)
        self.performance_history.append(objectives)

        sol = ParetoSolution(self.current_weights, objectives, reward)
        self.pareto_front.add_solution(sol)

    def update_weights(self, metrics=None, reward=None, day=None, step=None):
        """Update weights (compatibility)."""
        if reward is not None and metrics is not None:
            self.feedback(reward, metrics, day, step)
        return self.propose_weights()

    def get_current_weights(self):
        """
        Get current weight configuration.

        Returns:
            Dict[str, float]: Current weights.
        """
        return self.current_weights
