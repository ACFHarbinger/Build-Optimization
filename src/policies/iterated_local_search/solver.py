"""
Iterated Local Search (ILS) for Build Optimization.

ILS alternates between a local search descent phase and a perturbation phase.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import ILSParams


class ILSSolver(PolicyVizMixin):
    """
    Iterated Local Search solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: ILSParams,
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
        Run Iterated Local Search.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        for restart in range(self.params.n_restarts):
            if time.time() - start > self.params.time_limit:
                break

            # === Descent phase ===
            improved = True
            inner_count = 0
            while improved and inner_count < self.params.inner_iterations:
                if time.time() - start > self.params.time_limit:
                    break
                improved = False
                inner_count += 1

                llh = random.choice(self._llh_pool)
                try:
                    new_build = llh(current_build)
                    new_score = self.problem.evaluate(new_build)

                    if new_score > current_score + 1e-6:
                        current_build = new_build
                        current_score = new_score
                        improved = True

                        if current_score > best_score:
                            best_build = current_build.copy()
                            best_score = current_score
                except Exception:
                    continue

            # === Perturbation phase ===
            current_build = self._perturb(current_build)
            current_score = self.problem.evaluate(current_build)

            self._viz_record(
                iteration=restart,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
            )

        return best_build, best_score

    def _perturb(self, build: np.ndarray) -> np.ndarray:
        """Apply strong perturbation to escape local optimum."""
        n_remove = max(1, int(self.problem.num_slots * self.params.perturbation_strength))
        partial = random_removal(build, n_remove, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = worst_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = cluster_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)
