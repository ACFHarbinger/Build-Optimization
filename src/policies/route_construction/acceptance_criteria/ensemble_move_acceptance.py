"""
Ensemble Move Acceptance (EMA) Criterion.
"""

from typing import Any, Dict, List, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("ema")
class EnsembleAcceptance(IAcceptanceCriterion):
    """
    Meta-decision architecture combining multiple criteria.

    Evaluates a candidate solution through a portfolio of heterogeneous criteria
    and aggregates the decision via logical ensemble rules (e.g., G-AND, G-OR, G-VOT).
    """

    def __init__(self, criteria: List[IAcceptanceCriterion], rule: str = "G-VOT"):
        """
        Args:
            criteria (List[IAcceptanceCriterion]): The instantiated sub-criteria.
            rule (str): The aggregation logic ('G-AND', 'G-OR', 'G-VOT').
        """
        if not criteria:
            raise ValueError("EnsembleAcceptance requires at least one initialized criterion.")
        self.criteria = criteria
        self.rule = rule.upper()

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        for crit in self.criteria:
            crit.setup(initial_objective)

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Gather votes from all encapsulated criteria (List of Tuple[bool, AcceptanceMetrics])
        _results = [crit.accept(current_obj, candidate_obj) for crit in self.criteria]
        votes = [res[0] for res in _results]
        # Keep metrics for detailed analysis if needed (optional optimization)

        if self.rule == "G-AND":
            _accepted = bool(all(votes))  # Strict consensus
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}
        elif self.rule == "G-OR":
            _accepted = bool(any(votes))  # Authority rule (at least one)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}
        elif self.rule == "G-VOT":
            _accepted = bool(sum(votes) >= (len(votes) / 2.0))  # Majority rule
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj}
        else:
            raise ValueError(f"Unknown EMA rule: {self.rule}")

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Advance the state of all encapsulated criteria
        for crit in self.criteria:
            crit.step(current_obj, candidate_obj, accepted)

    def get_state(self) -> Dict[str, Any]:
        # Aggregate the states of the underlying criteria
        return {f"criterion_{i}": crit.get_state() for i, crit in enumerate(self.criteria)}
