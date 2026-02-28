"""
Reactive Tabu Search (RTS) for Build Optimization.

Uses hash-based cycle detection to dynamically adjust tabu tenure.
"""

import random
import time
from collections import deque
from typing import Deque, Dict, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_blink_insertion, greedy_insertion, regret_2_insertion
from .params import RTSParams


class RTSSolver(PolicyVizMixin):
    """
    Reactive Tabu Search solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: RTSParams,
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
        Run Reactive Tabu Search.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        tenure = self.params.initial_tenure
        # Tabu list: deque of solution hashes
        tabu_list: Deque[int] = deque(maxlen=self.params.max_tenure)
        # Hash history for cycle detection: hash -> last_seen_iteration
        hash_history: Dict[int, int] = {}
        no_repeat_count = 0

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Try a subset of LLHs or all if pool is small
            best_candidate = None
            best_cand_score = -float("inf")

            # We sample a few neighbors to find the "best" move
            for _ in range(5):
                llh = random.choice(self._llh_pool)
                try:
                    cand = llh(current_build)
                    cand_score = self.problem.evaluate(cand)
                    cand_hash = hash(cand.tobytes())

                    is_tabu = cand_hash in tabu_list

                    # Aspiration criterion: accept tabu if it's better than global best
                    if is_tabu and cand_score <= best_score:
                        continue

                    if cand_score > best_cand_score:
                        best_candidate = cand
                        best_cand_score = cand_score
                except Exception:
                    continue

            if best_candidate is None:
                # Diversify: force a random move even if it's potentially tabu
                llh = random.choice(self._llh_pool)
                try:
                    best_candidate = llh(current_build)
                    best_cand_score = self.problem.evaluate(best_candidate)
                except Exception:
                    continue

            if best_candidate is not None:
                current_build = best_candidate
                current_score = best_cand_score
                sol_hash = hash(current_build.tobytes())

                # Add to tabu list
                tabu_list.append(sol_hash)
                # Trim to current tenure
                while len(tabu_list) > tenure:
                    tabu_list.popleft()

                # Update global best
                if current_score > best_score:
                    best_build = current_build.copy()
                    best_score = current_score

                # Reactive tenure adjustment
                if sol_hash in hash_history:
                    # Cycle detected - increase tenure
                    tenure = min(self.params.max_tenure, int(tenure * self.params.tenure_increase) + 1)
                    no_repeat_count = 0
                else:
                    no_repeat_count += 1
                    if no_repeat_count > 2 * tenure:
                        # Long non-cycling - decrease tenure
                        tenure = max(self.params.min_tenure, int(tenure * self.params.tenure_decrease))
                        no_repeat_count = 0

                hash_history[sol_hash] = iteration

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                tenure=tenure,
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
