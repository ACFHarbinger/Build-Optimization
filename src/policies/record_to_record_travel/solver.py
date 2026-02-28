"""
Record-to-Record Travel (RR) for Build Optimization.

Accepts solutions within a decaying tolerance of the best-found record.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import RRParams


class RRSolver(PolicyVizMixin):
    """
    Record-to-Record Travel solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: RRParams,
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
        Run Record-to-Record Travel optimisation.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        record_build = current_build.copy()
        record_score = current_score

        # Tolerance band decays linearly
        initial_tolerance = self.params.tolerance * max(abs(record_score), 1.0)

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Linear decay
            progress = iteration / max(self.params.max_iterations - 1, 1)
            tolerance = initial_tolerance * (1.0 - progress)

            # Select and apply a random LLH
            llh = random.choice(self._llh_pool)
            try:
                new_build = llh(current_build)
                new_score = self.problem.evaluate(new_build)

                # RR acceptance
                if new_score >= record_score - tolerance:
                    current_build = new_build
                    current_score = new_score

                    # Update record
                    if current_score > record_score:
                        record_build = current_build.copy()
                        record_score = current_score
            except Exception:
                continue

            self._viz_record(
                iteration=iteration,
                best_profit=record_score,
                best_cost=-record_score,
                tolerance=tolerance,
            )

        return record_build, record_score

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = worst_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = cluster_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)
