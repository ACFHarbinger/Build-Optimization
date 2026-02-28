"""
Harmony Search (HS) algorithm for Build Optimization.

Models the optimisation process as a musical improvisation session.
New harmonies (builds) are created by picking slot values from
the Harmony Memory (HMCR) or random selection.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.repair_operators import greedy_insertion
from .params import HSParams


class HSSolver(PolicyVizMixin):
    """
    Harmony Search solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: HSParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run Harmony Search and return the best build found.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initialise Harmony Memory
        hm: List[np.ndarray] = [self.problem.random_solution() for _ in range(self.params.hm_size)]
        hm_scores = [self.problem.evaluate(h) for h in hm]

        best_idx = int(np.argmax(hm_scores))
        best_build = hm[best_idx].copy()
        best_score = hm_scores[best_idx]

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Improvise a new harmony
            new_harmony = self._improvise(hm)
            new_score = self.problem.evaluate(new_harmony)

            # Update HM: replace worst if new harmony is better
            worst_idx = int(np.argmin(hm_scores))
            if new_score > hm_scores[worst_idx]:
                hm[worst_idx] = new_harmony
                hm_scores[worst_idx] = new_score

                if new_score > best_score:
                    best_build = new_harmony.copy()
                    best_score = new_score

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
                hm_size=self.params.hm_size,
            )

        return best_build, best_score

    def _improvise(self, hm: List[np.ndarray]) -> np.ndarray:
        """
        Improvise a new build using HMCR and PAR.
        """
        new_build = np.full(self.problem.num_slots, -1, dtype=int)

        for slot in range(self.problem.num_slots):
            if random.random() < self.params.HMCR:
                # Pick from memory
                src_hm = random.choice(hm)
                selected_item = src_hm[slot]

                # Pitch adjustment
                if random.random() < self.params.PAR:
                    # Pick a different item for this slot
                    item_indices = np.where(self.problem.slot_ids == slot)[0]
                    if len(item_indices) > 0:
                        selected_item = random.choice(item_indices)
            else:
                # Random selection
                item_indices = np.where(self.problem.slot_ids == slot)[0]
                selected_item = random.choice(item_indices) if len(item_indices) > 0 else -1

            new_build[slot] = selected_item

        # Ensure feasibility and repair
        repaired = greedy_insertion(new_build, self.budget, self.problem)
        return repaired
