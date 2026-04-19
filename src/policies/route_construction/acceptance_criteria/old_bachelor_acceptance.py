"""
Old Bachelor Acceptance (OBA) Criterion.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("oba")
class OldBachelorAcceptance(IAcceptanceCriterion):
    """
    Dynamic, non-monotone threshold strategy.

    Tightens standards upon acceptance (contraction) and relaxes standards
    upon rejection streaks (dilation) to autonomously balance exploitation
    and exploration without a rigid decay schedule.
    """

    def __init__(self, contraction: float, dilation: float):
        """
        Args:
            contraction (float): Amount to reduce the threshold upon an accepted move.
            dilation (float): Base amount to increase the threshold upon a rejected move.
        """
        self.contraction = contraction
        self.dilation = dilation
        self.threshold = 0.0
        self.age = 0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        self.threshold = 0.0  # Start with strict greedy behavior
        self.age = 0

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # delta is negative for worsening moves.
        # e.g., candidate=90, current=100 -> delta=-10. If threshold is 15, -10 >= -15 (Accept).
        delta = candidate_obj - current_obj
        _accepted = bool(delta >= -self.threshold)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "threshold": self.threshold,
            "age": self.age,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted:
            # Move accepted: Reset rejection streak and tighten standards
            self.age = 0
            self.threshold = max(0.0, self.threshold - self.contraction)
        else:
            # Move rejected: Increase age and relax standards exponentially
            self.age += 1
            self.threshold += self.dilation * self.age

    def get_state(self) -> Dict[str, Any]:
        return {"threshold": self.threshold, "age": self.age}
