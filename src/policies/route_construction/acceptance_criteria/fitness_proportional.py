"""
Fitness Proportional Selection (FPS) Acceptance Criterion.
"""

import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("fp")
class FitnessProportionalAcceptance(IAcceptanceCriterion):
    """
    Accepts candidates probabilistically in proportion to their fitness relative
    to the current solution (similar to Roulette-Wheel selection adapted for
    binary choice).

    Probability of accepting the candidate:
    P(accept) = max(0, cand_obj) / (max(0, cur_obj) + max(0, cand_obj))
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Prevent negative fitness values
        cur_fit = max(0.0, current_obj)
        cand_fit = max(0.0, candidate_obj)

        total_fit = cur_fit + cand_fit
        if total_fit == 0.0:
            _accepted = bool(self.rng.random() < 0.5)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

        prob_accept = cand_fit / total_fit
        _accepted = bool(self.rng.random() < prob_accept)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
