"""
Great Deluge (GD) Acceptance Criterion.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("gd")
class GreatDelugeAcceptance(IAcceptanceCriterion):
    """
    Deterministic Great Deluge algorithm.

    Accepts a candidate if its objective is greater than or equal to a monotonically
    rising "water level". The water level rises linearly from the initial objective
    to a target objective over a predefined number of iterations.
    """

    def __init__(self, target_fitness_multiplier: float, max_iterations: int):
        """
        Args:
            target_fitness_multiplier (float): The expected relative improvement
                (e.g., 1.1 implies a 10% expected improvement over the initial solution).
            max_iterations (int): The total budget of iterations to reach the target.
        """
        self.target_multiplier = target_fitness_multiplier
        self.max_iterations = max_iterations
        self.water_level = 0.0
        self.rain_speed = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        self.water_level = initial_objective
        target_objective = initial_objective * self.target_multiplier

        # Calculate exactly how much the water should rise per step
        if self.max_iterations > 0:
            self.rain_speed = (target_objective - initial_objective) / self.max_iterations
        else:
            self.rain_speed = 0.0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj >= self.water_level)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "water_level": self.water_level,
            "rain_speed": self.rain_speed,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.water_level += self.rain_speed

    def get_state(self) -> Dict[str, Any]:
        return {"water_level": self.water_level, "rain_speed": self.rain_speed}
