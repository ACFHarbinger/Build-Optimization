import math
from typing import Any, Dict, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("nlgd")
class NonLinearGreatDeluge(IAcceptanceCriterion):
    """
    Non-Linear Great Deluge (NLGD) Acceptance Criterion (Landa-Silva et al., 2004).

    Implements a convex water-level decay that tracks the global best solution
    f_best with a small tolerance gap epsilon. This ensures exploration while
    forcing asymptotic convergence.

    Mathematical Formulation:
    Level_t = f_target + (Level_0 - f_target) * exp(-beta * t / t_max)
    where f_target = f_best * (1 - gap_epsilon) for minimization.
    """

    def __init__(
        self,
        t_max: int,
        initial_tolerance: float = 0.1,
        gap_epsilon: float = 0.01,
        beta: float = 5.0,
        maximization: bool = False,
    ):
        """
        Args:
            t_max (int): Total iteration budget.
            initial_tolerance (float): Factor for Level_0 relative to initial objective.
            gap_epsilon (float): Tolerance gap from f_best to prevent zero-distance stagnation.
            beta (float): Decay factor (convexity of the decay curve).
            maximization (bool): Whether the problem is maximization.
        """
        self.t_max = t_max
        self.initial_tolerance = initial_tolerance
        self.gap_epsilon = gap_epsilon
        self.beta = beta
        self.maximization = maximization

        self.level_0 = 0.0
        self.water_level = 0.0
        self.t = 0
        self.f_best_ever = float("inf") if not maximization else float("-inf")

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        if self.maximization:
            self.level_0 = initial_objective * (1.0 - self.initial_tolerance)
            self.f_best_ever = initial_objective
        else:
            self.level_0 = initial_objective * (1.0 + self.initial_tolerance)
            self.f_best_ever = initial_objective
        self.water_level = self.level_0
        self.t = 0

    def _get_f_best(self, **kwargs: Any) -> float:
        # User dynamic reference or internal tracking
        f_best = kwargs.get("f_best")
        if f_best is not None:
            if callable(f_best):
                return float(f_best())
            return float(f_best)
        return self.f_best_ever

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # f_best tracking for internal fallback
        if self.maximization:
            self.f_best_ever = max(self.f_best_ever, current_obj, candidate_obj)
        else:
            self.f_best_ever = min(self.f_best_ever, current_obj, candidate_obj)

        if self.maximization:
            _accepted = bool(candidate_obj >= self.water_level)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "water_level": self.water_level,
                "iteration": self.t,
                "progress": self.t / self.t_max if self.t_max > 0 else 0.0,
                "maximization": self.maximization,
            }
        else:
            _accepted = bool(candidate_obj <= self.water_level)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "water_level": self.water_level,
                "iteration": self.t,
                "progress": self.t / self.t_max if self.t_max > 0 else 0.0,
                "maximization": self.maximization,
            }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        self.t += 1
        f_best = self._get_f_best(**kwargs)

        # Calculate f_target with epsilon gap
        f_target = f_best * (1.0 + self.gap_epsilon) if self.maximization else f_best * (1.0 - self.gap_epsilon)

        # Non-linear decay schedule
        progress = min(1.0, self.t / self.t_max)
        decay = math.exp(-self.beta * progress)
        self.water_level = f_target + (self.level_0 - f_target) * decay

    def get_state(self) -> Dict[str, Any]:
        return {
            "water_level": self.water_level,
            "iteration": self.t,
            "progress": self.t / self.t_max if self.t_max > 0 else 0.0,
            "maximization": self.maximization,
        }
