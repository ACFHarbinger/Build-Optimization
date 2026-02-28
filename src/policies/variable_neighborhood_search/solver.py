"""
Variable Neighborhood Search (VNS) for Build Optimization.

VNS systematically changes neighborhood structures to escape local optima.
Uses shaking (destroy/repair) and local descent to explore the build space.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import VNSParams


class VNSSolver(PolicyVizMixin):
    """
    Variable Neighborhood Search solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: VNSParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        # Shaking neighborhoods N_1 ... N_{k_max}
        self._neighborhoods = [
            self._shake_n1,
            self._shake_n2,
            self._shake_n3,
            self._shake_n4,
            self._shake_n5,
        ]

        # LLH pool for local search descent phase
        self._llh_pool = [
            self._llh_greedy,
            self._llh_regret,
            self._llh_blink,
        ]

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run Variable Neighborhood Search.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()
        k_max = min(self.params.k_max, len(self._neighborhoods))

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            k = 0
            while k < k_max:
                if time.time() - start > self.params.time_limit:
                    break

                # === Shaking phase ===
                try:
                    shaken = self._neighborhoods[k](current_build.copy())
                except Exception:
                    k += 1
                    continue

                # === Local search descent phase ===
                ls_build, ls_score = self._local_search(shaken, start)

                # === Move or not (VNS acceptance criterion) ===
                if ls_score > current_score + 1e-6:
                    current_build = ls_build
                    current_score = ls_score
                    if current_score > best_score:
                        best_build = current_build.copy()
                        best_score = current_score
                    k = 0  # Improvement: restart from the mildest neighborhood
                else:
                    k += 1  # No improvement: try next neighborhood

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
            )

        return best_build, best_score

    # ------------------------------------------------------------------
    # Shaking neighborhoods
    # ------------------------------------------------------------------

    def _shake_n1(self, build: np.ndarray) -> np.ndarray:
        """N_1: Remove 1 slot randomly, greedy reinsert."""
        partial = random_removal(build, 1, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _shake_n2(self, build: np.ndarray) -> np.ndarray:
        """N_2: Remove 2 slots randomly, greedy reinsert."""
        partial = random_removal(build, 2, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _shake_n3(self, build: np.ndarray) -> np.ndarray:
        """N_3: Worst removal of 2 slots, regret-2 reinsert."""
        partial = worst_removal(build, 2, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _shake_n4(self, build: np.ndarray) -> np.ndarray:
        """N_4: Cluster removal of 3 slots, greedy reinsert."""
        partial = cluster_removal(build, 3, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _shake_n5(self, build: np.ndarray) -> np.ndarray:
        """N_5: Remove 3 slots randomly, regret-2 reinsert."""
        partial = random_removal(build, 3, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    # ------------------------------------------------------------------
    # Local search descent
    # ------------------------------------------------------------------

    def _local_search(self, build: np.ndarray, start_time: float) -> Tuple[np.ndarray, float]:
        """
        Apply repeated LLH improvement until no further progress.
        """
        score = self.problem.evaluate(build)
        current_build = build

        for _ in range(self.params.local_search_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            llh = random.choice(self._llh_pool)
            try:
                # Local search step: small removal and repair
                new_build = llh(current_build)
                new_score = self.problem.evaluate(new_build)

                if new_score > score + 1e-6:
                    current_build = new_build
                    score = new_score
            except Exception:
                continue

        return current_build, score

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)
