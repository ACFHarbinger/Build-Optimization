"""
Guided Local Search (GLS) for Build Optimization.

Augments the objective function with penalties on (slot, item) assignments
that frequently appear in local optima.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import GLSParams


class GLSSolver(PolicyVizMixin):
    """
    Guided Local Search solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: GLSParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        # Feature penalty matrix: penalties[slot_idx][item_idx]
        self.penalties = np.zeros((problem.num_slots, problem.num_items), dtype=np.float64)

        self._llh_pool = [
            self._llh_greedy,
            self._llh_regret,
            self._llh_blink,
        ]

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run GLS optimisation.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        for restart in range(self.params.max_restarts):
            if time.time() - start > self.params.time_limit:
                break

            # Inner local search loop using augmented objective
            for _ in range(self.params.inner_iterations):
                if time.time() - start > self.params.time_limit:
                    break

                llh = random.choice(self._llh_pool)
                try:
                    new_build = llh(current_build)
                    new_score = self.problem.evaluate(new_build)

                    # Accept if augmented objective improves
                    # We maximize (score - lambda * penalty)
                    aug_new = self._augmented_evaluate(new_build, new_score)
                    aug_cur = self._augmented_evaluate(current_build, current_score)

                    if aug_new >= aug_cur:
                        current_build = new_build
                        current_score = new_score

                        if current_score > best_score:
                            best_build = current_build.copy()
                            best_score = current_score
                except Exception:
                    continue

            # At local optimum: penalise the (slot, item) features with highest utility
            self._update_penalties(current_build)

            self._viz_record(
                iteration=restart,
                best_profit=best_score,
                best_cost=-best_score,
            )

        return best_build, best_score

    def _update_penalties(self, build: np.ndarray) -> None:
        """Penalise the (slot, item) features with highest utility in current build."""
        best_utility = -1.0
        best_features = []

        for slot_idx, item_idx in enumerate(build):
            if item_idx == -1:
                continue

            # Utility = Feature_Cost / (1 + Penalty)
            # In Build models, "cost" of a feature is the item cost
            cost = self.problem.costs[item_idx]
            utility = cost / (1.0 + self.penalties[slot_idx][item_idx])

            if utility > best_utility:
                best_utility = utility
                best_features = [(slot_idx, item_idx)]
            elif abs(utility - best_utility) < 1e-9:
                best_features.append((slot_idx, item_idx))

        # Penalize one or all features with max utility
        for s_idx, i_idx in best_features:
            self.penalties[s_idx][i_idx] += 1.0

    def _augmented_evaluate(self, build: np.ndarray, score: float) -> float:
        """Evaluate with penalty-augmented objective."""
        penalty_sum = 0.0
        for slot_idx, item_idx in enumerate(build):
            if item_idx != -1:
                penalty_sum += self.penalties[slot_idx][item_idx]

        return score - self.params.lambda_param * penalty_sum

    def _llh_greedy(self, build: np.ndarray) -> np.ndarray:
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _llh_regret(self, build: np.ndarray) -> np.ndarray:
        partial = worst_removal(build, self.params.n_removal, self.problem)
        return regret_2_insertion(partial, self.budget, self.problem)

    def _llh_blink(self, build: np.ndarray) -> np.ndarray:
        partial = cluster_removal(build, self.params.n_removal, self.problem)
        return greedy_blink_insertion(partial, self.budget, self.problem)
