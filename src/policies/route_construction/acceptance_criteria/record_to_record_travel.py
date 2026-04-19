"""
Record-to-Record Travel (RRT) Criterion.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("rrt")
class RecordToRecordTravel(IAcceptanceCriterion):
    """
    Deterministic Record-to-Record Travel.

    Accepts a candidate solution if its objective is within a defined percentage
    tolerance (deviation) of the GLOBAL best known solution, rather than the
    immediate predecessor.
    """

    def __init__(self, tolerance: float):
        """
        Args:
            tolerance (float): The allowed percentage deviation from the global best
                (e.g., 0.05 means solutions within 5% of the global best are accepted).
        """
        self.tolerance = tolerance
        self.global_best = -float("inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        self.global_best = initial_objective

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Assuming strictly positive profit objectives.
        # Accept if candidate is greater than or equal to 95% of the global best.
        allowable_floor = self.global_best * (1.0 - self.tolerance)
        _accepted = bool(candidate_obj >= allowable_floor)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "global_best": self.global_best,
            "tolerance_floor": self.global_best * (1.0 - self.tolerance),
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Update the global best record if the new solution dominates it
        if accepted and current_obj > self.global_best:
            self.global_best = current_obj

    def get_state(self) -> Dict[str, Any]:
        return {"global_best": self.global_best, "tolerance_floor": self.global_best * (1.0 - self.tolerance)}
