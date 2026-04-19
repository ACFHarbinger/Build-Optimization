import math
import random
from collections import deque
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion, ObjectiveValue
from logic.src.interfaces.context.search_context import AcceptanceMetrics

from .base.registry import AcceptanceCriterionRegistry


@AcceptanceCriterionRegistry.register("abm")
class AdaptiveBoltzmannMetropolis(IAcceptanceCriterion):
    """
    Adaptive Boltzmann Metropolis Criterion.

    Statistically rigorously scales the temperature based on the search landscape's
    topography. Tracks the standard deviation of objective variations (sigma_delta_f)
    to dynamically adjust the cooling schedule and initial temperature.

    Formula for T_0:
    T_0 = -sigma_delta_f / ln(p_0)
    where p_0 is the desired probability of accepting a 1-sigma regressing move.
    """

    def __init__(
        self,
        p0: float = 0.5,
        window_size: int = 100,
        alpha: float = 0.95,
        min_temp: float = 1e-6,
        seed: Optional[int] = None,
        maximization: bool = True,
    ):
        """
        Args:
            p0 (float): Initial target probability for accepting a 1-sigma move.
            window_size (int): Size of the moving window for statistical tracking.
            alpha (float): Cooling rate factor (geometric cooling).
            min_temp (float): Lower bound for temperature.
            seed (Optional[int]): Random seed.
            maximization (bool): Whether the problem is maximization.
        """
        self.p0 = p0
        self.window_size = window_size
        self.alpha = alpha
        self.min_temp = min_temp
        self.maximization = maximization

        self.rng = random.Random(seed)
        self.deltas: deque[float] = deque(maxlen=window_size)

        self.temp = 0.0  # Initialized during first transitions
        self.sigma = 0.0

    def setup(self, initial_objective: ObjectiveValue) -> None:
        initial_objective = cast(float, initial_objective)
        pass

    def _update_stats(self, delta: float) -> None:
        self.deltas.append(abs(delta))
        if len(self.deltas) >= 5:  # Need a minimum window to compute sigma
            self.sigma = float(np.std(self.deltas))
            if self.sigma > 0 and self.temp == 0.0:
                # Initialize T0 based on first observed sigma
                self.temp = -self.sigma / math.log(self.p0)

    def accept(
        self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, **kwargs: Any
    ) -> Tuple[bool, AcceptanceMetrics]:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if not self.maximization:
            delta = -delta

        # Always accept improving or equal moves
        if delta >= 0:
            _accepted = bool(True)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "sigma": self.sigma,
                "window_len": len(self.deltas),
            }

        # If temperature is not yet initialized, reject worsening
        if self.temp <= self.min_temp:
            _accepted = bool(False)
            return _accepted, {
                "accepted": _accepted,
                "delta": candidate_obj - current_obj,
                "temperature": self.temp,
                "sigma": self.sigma,
                "window_len": len(self.deltas),
            }

        # Boltzmann probability: P = exp(delta / T)
        # Note: delta is negative here for worsening moves
        prob = math.exp(delta / self.temp)
        _accepted = bool(self.rng.random() < prob)
        return _accepted, {
            "accepted": _accepted,
            "delta": candidate_obj - current_obj,
            "temperature": self.temp,
            "sigma": self.sigma,
            "window_len": len(self.deltas),
        }

    def step(self, current_obj: ObjectiveValue, candidate_obj: ObjectiveValue, accepted: bool, **kwargs: Any) -> None:
        current_obj = cast(float, current_obj)
        candidate_obj = cast(float, candidate_obj)
        delta = candidate_obj - current_obj
        if not self.maximization:
            delta = -delta

        self._update_stats(delta)

        # Basic cooling schedule
        if self.temp > self.min_temp:
            self.temp *= self.alpha

    def get_state(self) -> Dict[str, Any]:
        return {"temperature": self.temp, "sigma": self.sigma, "window_len": len(self.deltas)}
