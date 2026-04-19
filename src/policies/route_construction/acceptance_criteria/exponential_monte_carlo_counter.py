import random
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("emc_q")
class EMCQAcceptance(IAcceptanceCriterion):
    """
    Exponential Monte Carlo with Counter (EMC-Q) Acceptance Criterion.
    Reference: Franzin & Stützle (2019).

    Tracks consecutive rejections. After Q failures, the fixed probability p
    is replaced by a temporary boost, preventing indefinite stagnation on flat regions.

    Mathematical Formulation:
    P(A | worsening) = p        if q < Q
    P(A | worsening) = p_boost  if q >= Q
    where q is the consecutive rejection counter and Q is the threshold.
    """

    def __init__(
        self,
        p: float = 0.05,
        p_boost: float = 0.5,
        q_threshold: Union[int, Callable[[], int]] = 100,
        seed: Optional[int] = None,
        maximization: bool = False,
    ):
        """
        Args:
            p (float): Baseline acceptance probability for worsening moves.
            p_boost (float): Boosted acceptance probability after reaching threshold Q.
            q_threshold (Union[int, Callable]): The rejection counter threshold Q.
                Must be an int or a callable returning an int.
            seed (Optional[int]): Random seed.
            maximization (bool): Whether the problem is maximization.
        """
        self.p = p
        self.p_boost = p_boost
        self.q_threshold = q_threshold
        self.maximization = maximization

        self.rng = random.Random(seed)
        self.rejection_counter = 0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        self.rejection_counter = 0

    def _get_q_threshold(self) -> int:
        if callable(self.q_threshold):
            return int(self.q_threshold())
        return int(self.q_threshold)

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # Always accept improving or equal moves
        if delta <= 0:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "rejection_counter": self.rejection_counter,
                "q_threshold": self._get_q_threshold(),
                "p_active": self.p if self.rejection_counter < self._get_q_threshold() else self.p_boost,
                "maximization": self.maximization,
            }

        # Worsening move: check counter q against threshold Q
        q_thresh = self._get_q_threshold()
        prob = self.p if self.rejection_counter < q_thresh else self.p_boost

        _accepted = bool(self.rng.random() < prob)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "rejection_counter": self.rejection_counter,
            "q_threshold": self._get_q_threshold(),
            "p_active": self.p if self.rejection_counter < self._get_q_threshold() else self.p_boost,
            "maximization": self.maximization,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        if accepted:
            # Reset counter on any acceptance (improving or worsening)
            self.rejection_counter = 0
        else:
            # Increment on rejection
            self.rejection_counter += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "rejection_counter": self.rejection_counter,
            "q_threshold": self._get_q_threshold(),
            "p_active": self.p if self.rejection_counter < self._get_q_threshold() else self.p_boost,
            "maximization": self.maximization,
        }
