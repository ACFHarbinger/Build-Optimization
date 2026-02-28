"""
Local Search Base Module for Build Optimization.

This module defines the abstract base class for local search algorithms
operating on discrete item-slot builds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin


class LocalSearch(PolicyVizMixin, ABC):
    """
    Abstract base class for Local Search algorithms in Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: Any,
    ):
        """
        Initialize Local Search base class.

        Args:
            problem: BuildProblem instance.
            budget: Maximum cost budget.
            params: Parameters for the local search.
        """
        self.problem = problem
        self.budget = budget
        self.params = params

    @abstractmethod
    def optimize(self, build: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Optimize the given build.
        """
        pass

    def _evaluate(self, build: np.ndarray) -> float:
        """Score a build."""
        return self.problem.evaluate(build)

    def _is_feasible(self, build: np.ndarray) -> bool:
        """Check budget feasibility."""
        return self.problem.is_feasible(build)
