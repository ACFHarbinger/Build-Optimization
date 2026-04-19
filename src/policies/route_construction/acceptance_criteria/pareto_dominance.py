from typing import Any, Dict, Sequence, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("pd")
class ParetoDominanceAcceptance(IAcceptanceCriterion):
    """
    Accepts a candidate only if it strictly dominates the current solution across
    all objectives (Pareto superiority).

    This expects the `current_obj` and `candidate_obj` parameters to actually be
    iterables of floats (e.g., tuple or list of objective values) rather than scalar floats.
    """

    def setup(self, initial_objective: ObjectiveValue) -> None:
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        # Multi-objective criteria expect Sequence[float]
        cur_objs = cast(Sequence[float], current_obj)
        cand_objs = cast(Sequence[float], candidate_obj)

        if len(cur_objs) != len(cand_objs):
            raise ValueError("Objective dimensions must match for Pareto dominance.")

        # Candidate dominates if it is >= in all objectives, and > in at least one
        # Assuming maximization for all objectives (as per VRPP structure)
        better_in_one = False
        for cur_val, cand_val in zip(cur_objs, cand_objs, strict=True):
            if cand_val < cur_val:
                _accepted = bool(False)
                return _accepted, {"accepted": _accepted, "delta": 0.0}
            if cand_val > cur_val:
                better_in_one = True

        _accepted = bool(better_in_one)
        return _accepted, {"accepted": _accepted, "delta": 0.0}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
