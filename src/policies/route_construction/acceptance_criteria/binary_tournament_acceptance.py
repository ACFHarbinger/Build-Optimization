"""
Simulated Tournament Acceptance (STA) Criterion.
"""

import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("bta")
class BinaryTournamentAcceptance(IAcceptanceCriterion):
    """
    Binary Tournament stochastic acceptance.
    Evaluates the current solution against the candidate solution. The superior
    solution "wins" the tournament with probability `p` (where `p` > 0.5), and
    the inferior solution wins with probability `1 - p`.

    If the candidate wins, it is accepted.
    """

    def __init__(self, p: float = 0.8, seed: Optional[int] = None) -> None:
        """
        Args:
            p (float): The probability that the superior solution wins the tournament.
                Typically between 0.5 and 1.0.
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.p = p
        self.rng = random.Random(seed)

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        roll = self.rng.random()
        candidate_is_superior = candidate_obj > current_obj

        if candidate_is_superior:
            # Candidate gets the high probability
            _accepted = bool(roll < self.p)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "tournament_probability": self.p,
            }
        else:
            # Candidate is worse, so it gets the low probability
            _accepted = bool(roll > self.p)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "tournament_probability": self.p,
            }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"tournament_probability": self.p}
