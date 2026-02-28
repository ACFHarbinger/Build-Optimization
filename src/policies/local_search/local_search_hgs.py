"""
HGS Local Search Module for Build Optimization.

Applies local refinement to HGS Individuals (builds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

from .local_search_base import LocalSearch

if TYPE_CHECKING:
    from ..hybrid_genetic_search.individual import Individual
    from ..hybrid_genetic_search.params import HGSParams


class HGSLocalSearch(LocalSearch):
    """
    Local Search module for HGS.
    """

    def __init__(
        self,
        problem: Any,
        budget: float,
        params: HGSParams,
    ):
        super().__init__(problem, budget, params)

    def optimize(self, build: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Refine a build using local search.
        """
        # In build optimization, HGS local search refines the discrete selection.
        # This is a simplified version that ensures score is updated.
        new_build = build.copy()

        # Optional: Apply some hill climbing or swaps here.
        # For now, it's a placeholder that evaluates the build.

        score = self.problem.evaluate(new_build)
        return new_build, score

    def optimize_individual(self, individual: Individual) -> Individual:
        """
        Refine an individual's build using local search.
        """
        new_build, score = self.optimize(individual.build)
        individual.build = new_build
        individual.score = score
        individual.is_feasible = self.problem.is_feasible(new_build)
        return individual
