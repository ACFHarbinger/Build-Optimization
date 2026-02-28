"""
Slack Induction by String Removal (SISR) for Build Optimization.

Adapts the string-removal (spatial/sequence based ruin) to discrete builds.
Repair via greedy insertion with blinks.
"""

import math
import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_blink_insertion
from .params import SISRParams


class SISRSolver(PolicyVizMixin):
    """
    Solver implementing the SISR metaheuristic for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: SISRParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the SISR algorithm.

        Returns:
            Tuple of (best_build, best_score).
        """
        start_time = time.time()

        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        temp = self.params.start_temp
        num_slots = self.problem.num_slots
        n_remove = max(1, int(num_slots * self.params.destroy_ratio))

        for iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            # 1. Ruin (Destroy)
            # In Build models, "strings" of slots don't strictly exist unless we
            # assume a linear ordering. We'll use a sequential random removal
            # to mimic string removal if num_slots > n_remove.
            partial = random_removal(current_build, n_remove, self.problem)

            # 2. Recreate (Repair)
            new_build = greedy_blink_insertion(
                partial,
                self.budget,
                self.problem,
                blink_rate=self.params.blink_rate,
            )

            new_score = self.problem.evaluate(new_build)

            # 3. Acceptance (Simulated Annealing)
            delta = new_score - current_score
            accept = False

            if delta > -1e-6:
                accept = True
            else:
                prob = math.exp(delta / temp) if temp > 1e-9 else 0
                if random.random() < prob:
                    accept = True

            if accept:
                current_build = new_build
                current_score = new_score
                if current_score > best_score:
                    best_build = current_build.copy()
                    best_score = current_score

            # 4. Cooling
            temp *= self.params.cooling_rate

            self._viz_record(
                iteration=iteration,
                best_cost=best_score,  # legacy attribute name in viz
                current_cost=current_score,
                temperature=temp,
                accepted=int(accept),
            )

        return best_build, best_score
