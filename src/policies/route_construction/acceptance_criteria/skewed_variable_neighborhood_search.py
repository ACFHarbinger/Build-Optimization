from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics
from logic.src.interfaces.distance_metric import IDistanceMetric

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("svns")
class SkewedVNSAcceptance(IAcceptanceCriterion):
    """
    Skewed Variable Neighborhood Search (SVNS) Acceptance Criterion.

    Accepts a candidate solution if it is either better than the current solution
    or if it is sufficiently distant (structurally diverse) to justify a slight
    objective deterioration. This prevents the search from being trapped in
    isomorphic basins.

    Mathematical Formulation (Maximization):
    Accept if: f_cand > f_cur - alpha * rho(x_cur, x_cand)
    where rho is the structural distance calculated by an IDistanceMetric.
    """

    def __init__(self, alpha: float, metric: IDistanceMetric, maximization: bool = True):
        """
        Args:
            alpha (float): Scaling factor for structural distance.
            metric (IDistanceMetric): Metric for computing structural distance rho.
            maximization (bool): Whether the problem is a maximization problem.
        """
        self.alpha = alpha
        self.metric = metric
        self.maximization = maximization

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        """
        Args:
            current_obj (float): Objective of the current solution.
            candidate_obj (float): Objective of the candidate solution.
            **kwargs: Must contain 'current_sol' and 'candidate_sol' for distance calculation.
        """
        if self.maximization:
            if candidate_obj > current_obj:
                _accepted = bool(True)
                return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}
        else:
            if candidate_obj < current_obj:
                _accepted = bool(True)
                return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

        current_sol = kwargs.get("current_sol")
        candidate_sol = kwargs.get("candidate_sol")

        if current_sol is None or candidate_sol is None:
            # Fallback to only improving if structural data is missing
            _accepted = bool(False)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

        rho = self.metric.compute(current_sol, candidate_sol)

        if self.maximization:
            _accepted = bool(candidate_obj > (current_obj - self.alpha * rho))
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}
        else:
            _accepted = bool(candidate_obj < (current_obj + self.alpha * rho))
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "alpha": self.alpha}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        pass

    def get_state(self) -> Dict[str, Any]:
        return {"alpha": self.alpha}
