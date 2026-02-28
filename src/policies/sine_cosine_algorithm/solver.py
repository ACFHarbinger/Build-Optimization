"""
Sine Cosine Algorithm (SCA) for Build Optimization.

Updates positions using trigonometric wave functions.
Decodes continuous vectors to discrete builds via ranking and greedy filling.
"""

import math
import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .params import SCAParams


class SCASolver(PolicyVizMixin):
    """
    Sine Cosine Algorithm solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: SCAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run SCA and return the best build found.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()
        T = self.params.max_iterations
        num_slots = self.problem.num_slots

        # Initialise population in continuous space
        X = np.random.uniform(-1.0, 1.0, (self.params.pop_size, num_slots))
        builds_pop = [self._decode(x) for x in X]
        scores = [self.problem.evaluate(b) for b in builds_pop]

        best_idx = int(np.argmax(scores))
        X_best = X[best_idx].copy()
        best_build = builds_pop[best_idx].copy()
        best_score = scores[best_idx]

        for t in range(T):
            if time.time() - start > self.params.time_limit:
                break

            # Control parameter decays from a_max → 0
            a = self.params.a_max * (1.0 - t / T)

            for i in range(self.params.pop_size):
                r1 = random.uniform(0, a)
                r2 = random.uniform(0, 2 * math.pi)
                r3 = random.uniform(0, 2)
                r4 = random.random()

                diff = r3 * X_best - X[i]

                if r4 < 0.5:
                    X[i] = X[i] + r1 * math.sin(r2) * np.abs(diff)
                else:
                    X[i] = X[i] + r1 * math.cos(r2) * np.abs(diff)

                # Decode and evaluate
                builds_pop[i] = self._decode(X[i])
                scores[i] = self.problem.evaluate(builds_pop[i])

                if scores[i] > best_score:
                    X_best = X[i].copy()
                    best_build = builds_pop[i].copy()
                    best_score = scores[i]

            self._viz_record(
                iteration=t,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
                a=a,
            )

        return best_build, best_score

    def _decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decode a continuous position vector to a discrete build.
        """
        ranked_slots = np.argsort(x)[::-1]

        build = np.full(self.problem.num_slots, -1, dtype=int)
        current_cost = 0.0

        for slot in ranked_slots:
            item_indices = np.where(self.problem.slot_ids == slot)[0]
            if len(item_indices) == 0:
                continue

            best_item = -1
            best_val = -1e9

            for item_idx in item_indices:
                cost = self.problem.costs[item_idx]
                if current_cost + cost <= self.budget:
                    score = self.problem.yields[item_idx] / (cost + 1e-6)
                    if score > best_val:
                        best_val = score
                        best_item = item_idx

            if best_item != -1:
                build[slot] = best_item
                current_cost += self.problem.costs[best_item]

        return build
