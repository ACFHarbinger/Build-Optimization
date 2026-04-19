import math
import random
from collections import deque
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("gt_sa")
class GeneralizedTsallisSA(IAcceptanceCriterion):
    """
    Generalized (Tsallis) Simulated Annealing Acceptance Criterion (Tsallis & Stariolo, 1996).

    Uses a non-extensive entropy formulation to generate heavy-tailed acceptance
    probabilities, allowing for larger jumps out of deep local minima compared to
    standard Boltzmann-Gibbs SA.

    Mathematical Formulation:
    P(A) = [1 - (1-q) * (delta_f / T)]^(1 / (1-q))
    where q is the non-extensivity parameter (q=1 converges to standard SA).
    """

    def __init__(
        self,
        q: float = 1.5,
        p0: float = 0.5,
        window_size: int = 100,
        alpha: float = 0.95,
        min_temp: float = 1e-6,
        seed: Optional[int] = None,
        maximization: bool = False,
    ):
        """
        Args:
            q (float): Non-extensivity parameter. Must be in (1, 3).
            p0 (float): Target probability for accepting a 1-sigma worsening move at T0.
            window_size (int): Moving window size for sigma calculation.
            alpha (float): Cooling rate factor.
            min_temp (float): Minimum allowable temperature.
            seed (Optional[int]): Random seed.
            maximization (bool): Whether the problem is maximization.
        """
        if not (1.0 < q < 3.0):
            raise ValueError("Parameter q must be in the range (1, 3) for Tsallis SA.")

        self.q = q
        self.p0 = p0
        self.window_size = window_size
        self.alpha = alpha
        self.min_temp = min_temp
        self.maximization = maximization

        self.rng = random.Random(seed)
        self.deltas: deque[float] = deque(maxlen=window_size)
        self.temp = 0.0
        self.sigma = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def _update_stats(self, delta: float) -> None:
        self.deltas.append(abs(delta))
        if len(self.deltas) >= 5 and self.temp == 0.0:
            self.sigma = float(np.std(self.deltas))
            if self.sigma > 1e-9:
                # Calculate T0 such that P(sigma) = p0
                # T0 = (1-q) * sigma / (1 - p0^(1-q))
                numerator = (1.0 - self.q) * self.sigma
                denominator = 1.0 - math.pow(self.p0, 1.0 - self.q)
                self.temp = numerator / denominator

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        # Always accept improving or equal moves (delta <= 0)
        if delta <= 0:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "q": self.q,
                "sigma": self.sigma,
                "maximization": self.maximization,
            }

        if self.temp <= self.min_temp:
            _accepted = bool(False)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "q": self.q,
                "sigma": self.sigma,
                "maximization": self.maximization,
            }

        # Tsallis probability: [1 - (1-q) * (delta / T)]^(1/(1-q))
        term = 1.0 - (1.0 - self.q) * (delta / self.temp)
        if term <= 0:
            _accepted = bool(False)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "q": self.q,
                "sigma": self.sigma,
                "maximization": self.maximization,
            }

        prob = math.pow(term, 1.0 / (1.0 - self.q))
        _accepted = bool(self.rng.random() < prob)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "temperature": self.temp,
            "q": self.q,
            "sigma": self.sigma,
            "maximization": self.maximization,
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if self.maximization:
            delta = -delta

        self._update_stats(delta)

        # Apply cooling schedule
        if self.temp > self.min_temp:
            self.temp *= self.alpha

    def get_state(self) -> Dict[str, Any]:
        return {
            "temperature": self.temp,
            "q": self.q,
            "sigma": self.sigma,
            "maximization": self.maximization,
        }
