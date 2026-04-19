"""
Monte Carlo Acceptance (MCA) Criterion.
"""

import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("mc")
class MonteCarloAcceptance(IAcceptanceCriterion):
    """
    Fixed-probability stochastic acceptance.
    Deterministically accepts improving candidates.
    Worsening candidates are accepted with a fixed, unannealed probability `p`.
    """

    def __init__(self, p: float = 0.1, seed: Optional[int] = None) -> None:
        """
        Args:
            p (float): The probability of accepting a worsening move (0.0 to 1.0).
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.p = max(0.0, min(1.0, p))
        self.rng = random.Random(seed)

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if candidate_obj >= current_obj:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "acceptance_probability": self.p,
            }
        _accepted = bool(self.rng.random() < self.p)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "acceptance_probability": self.p,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"acceptance_probability": self.p}
