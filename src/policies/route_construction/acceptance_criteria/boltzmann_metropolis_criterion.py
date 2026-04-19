"""
Boltzmann-Metropolis Criterion (BMC) / Simulated Annealing.
"""

import math
import random
from typing import Any, Dict, Optional, Tuple, cast

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("bmc")
class BoltzmannAcceptance(IAcceptanceCriterion):
    """
    Metropolis-Boltzmann acceptance criterion governed by a geometric cooling schedule.

    Improving moves are accepted deterministically. Worsening moves are accepted
    with probability P = exp(Δf / T), where Δf < 0 for maximization.
    """

    def __init__(self, initial_temp: float, alpha: float, seed: Optional[int] = 42):
        """
        Args:
            initial_temp (float): The starting temperature parameter.
            alpha (float): The geometric cooling rate (typically 0.90 - 0.995).
            seed (Optional[int]): Random seed for stochastic reproducibility.
        """
        self.T = initial_temp
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
        delta = candidate_obj - current_obj
        if delta >= 0:
            _accepted = bool(True)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "temperature": self.T}

        # delta is negative, T is positive -> exp(negative) gives valid [0, 1] probability
        if self.T <= 1e-9:
            _accepted = bool(False)
            return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "temperature": self.T}

        _accepted = bool(self.rng.random() < math.exp(delta / self.T))
        return _accepted, {"accepted": _accepted, "delta": candidate_obj - current_obj, "temperature": self.T}

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        # Geometric decay
        self.T *= self.alpha

    def get_state(self) -> Dict[str, Any]:
        return {"temperature": self.T}
