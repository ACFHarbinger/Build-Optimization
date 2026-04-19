"""
Threshold Accepting (TA) Criterion.
"""

from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ta")
class ThresholdAccepting(IAcceptanceCriterion):
    """
    Deterministic Threshold Accepting.

    A deterministic counterpart to Simulated Annealing. It explicitly bounds the
    allowable deterioration using a threshold parameter that decays linearly
    towards zero over the specified iteration budget.
    """

    def __init__(self, initial_threshold: float, max_iterations: int):
        """
        Args:
            initial_threshold (float): The starting absolute tolerance for deterioration.
            max_iterations (int): The budget used to calculate linear decay.
        """
        self.threshold = initial_threshold
        self.decay_rate = initial_threshold / max_iterations if max_iterations > 0 else 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # delta = candidate - current. If worsening by 10, delta is -10.
        # If threshold is 15, -10 >= -15 is True.
        delta = candidate_obj - current_obj
        _accepted = bool(delta >= -self.threshold)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "threshold": self.threshold}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.threshold = max(0.0, self.threshold - self.decay_rate)

    def get_state(self) -> Dict[str, Any]:
        return {"threshold": self.threshold}
