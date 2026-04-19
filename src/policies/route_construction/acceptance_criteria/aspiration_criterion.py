"""
Aspiration Criterion (AC).
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ac")
class AspirationCriterion(IAcceptanceCriterion):
    """
    Strictest form of memory-overriding acceptance.
    Accepts a candidate only if its objective surpasses the global best objective
    found since the criterion was initialized.
    """

    def __init__(self) -> None:
        self.global_best = -float("inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        self.global_best = initial_objective

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        _accepted = bool(candidate_obj > self.global_best)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "global_best_objective": self.global_best,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted and candidate_obj > self.global_best:
            self.global_best = candidate_obj

    def get_state(self) -> Dict[str, Any]:
        return {"global_best_objective": self.global_best}
