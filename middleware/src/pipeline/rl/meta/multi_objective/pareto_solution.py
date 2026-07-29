"""pareto_solution.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import pareto_solution
"""

import copy


class ParetoSolution:
    """Represents a solution on the Pareto front."""

    def __init__(self, weights, objectives, reward, model_id=None):
        """
        Initialize a Pareto solution.

        Args:
            weights: Weight configuration dict.
            objectives: Objective values dict.
            reward: Total reward.
            model_id: Optional model identifier.
        """
        self.weights = copy.deepcopy(weights)
        self.objectives = copy.deepcopy(objectives)
        self.reward = reward
        self.model_id = model_id

    def dominates(self, other):
        """
        Check if this solution dominates another.

        Args:
            other: Another ParetoSolution.

        Returns:
            bool: True if this solution dominates the other.
        """
        # Higher is better for all objectives assumed here (should be normalized)
        # Or specifically handled for waste_efficiency (max) and overflow_rate (min)
        waste_better = self.objectives.get("waste_efficiency", 0) >= other.objectives.get("waste_efficiency", 0)
        overflow_better = self.objectives.get("overflow_rate", 1) <= other.objectives.get("overflow_rate", 1)

        strictly_better = (
            self.objectives.get("waste_efficiency", 0) > other.objectives.get("waste_efficiency", 0)
        ) or (self.objectives.get("overflow_rate", 1) < other.objectives.get("overflow_rate", 1))

        return waste_better and overflow_better and strictly_better
