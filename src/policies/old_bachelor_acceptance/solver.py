"""
Old Bachelor Acceptance (OBA) for Build Optimization.

Threshold dilates after rejections and contracts after acceptances.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import OBAParams


class OBASolver(PolicyVizMixin):
    """
    Old Bachelor Acceptance solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: OBAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        self._llh_pool = [
            self._llh_greedy,
            self._llh_regret,
            self._llh_blink,
        ]

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run OBA optimisation.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        # OBA threshold
        threshold = 0.0

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Select and apply a random LLH
            llh = random.choice(self._llh_pool)
            try:
                new_build = llh(current_build)
                new_score = self.problem.evaluate(new_build)

                # OBA acceptance
                if new_score >= current_score - threshold:
                    current_build = new_build
                    current_score = new_score

                    # Contract threshold on acceptance
                    threshold = max(0.0, threshold - self.params.contraction)

                    if current_score > best_score:
                        best_build = current_build.copy()
                        best_score = current_score
                else:
                    # Dilate threshold on rejection
                    threshold += self.params.dilation
            except Exception:
                threshold += self.params.dilation
                continue

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                threshold=threshold,
            )

        return best_build, best_score

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = worst_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = cluster_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)
