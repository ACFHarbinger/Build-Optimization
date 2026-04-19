"""
Probabilistic Transition Acceptance Criterion.
"""

import math
import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("pt")
class ProbabilisticTransitionAcceptance(IAcceptanceCriterion):
    """
    Acceptance criterion inspired by Ant Colony Optimization (ACO) transition rules.
    It evaluates the attractiveness of the candidate relative to the current state,
    using an exponentiated formulation.

    P(accept) = (cand_obj^alpha) / (cur_obj^alpha + cand_obj^alpha)
    """

    def __init__(self, alpha: float = 1.0, seed: Optional[int] = None) -> None:
        """
        Args:
            alpha (float): Scaling factor governing how aggressively higher objectives
                are preferred. High alpha is more greedy, low alpha is more random.
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.alpha = alpha
        self.rng = random.Random(seed)

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        cur_fit = max(0.0, current_obj)
        cand_fit = max(0.0, candidate_obj)

        if cur_fit == 0.0 and cand_fit == 0.0:
            _accepted = bool(self.rng.random() < 0.5)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

        cur_weight = math.pow(cur_fit, self.alpha) if cur_fit > 0 else 0.0
        cand_weight = math.pow(cand_fit, self.alpha) if cand_fit > 0 else 0.0

        prob_accept = cand_weight / (cur_weight + cand_weight)
        _accepted = bool(self.rng.random() < prob_accept)
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"alpha": self.alpha}
