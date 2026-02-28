"""
Discrete Firefly Algorithm (DFA) for Build Optimization.

Adapts the FA to discrete builds by using Hamming distance and
probabilistic slot adoption governed by attractiveness (beta).
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_insertion
from .params import FAParams


class FASolver(PolicyVizMixin):
    """
    Discrete Firefly Algorithm solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: FAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the Discrete Firefly Algorithm.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        population = [self.problem.random_solution() for _ in range(self.params.pop_size)]
        scores = [self.problem.evaluate(f) for f in population]

        best_idx = int(np.argmax(scores))
        best_build = population[best_idx].copy()
        best_score = scores[best_idx]

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Pairwise attraction: dimmer firefly moves toward brighter one
            for i in range(self.params.pop_size):
                moved = False
                for j in range(self.params.pop_size):
                    if scores[j] <= scores[i]:
                        continue

                    # Hamming distance
                    d = np.sum(population[i] != population[j])
                    beta = self.params.beta0 * np.exp(-self.params.gamma * d * d)

                    if random.random() < beta:
                        new_build = self._attract(population[i], population[j])
                        if self.problem.is_feasible(new_build):
                            new_score = self.problem.evaluate(new_build)
                            if new_score > scores[i]:
                                population[i] = new_build
                                scores[i] = new_score
                                moved = True

                # Random walk if not attracted or by chance
                if not moved or random.random() < self.params.alpha_rnd:
                    rw = self._random_walk(population[i])
                    rw_score = self.problem.evaluate(rw)
                    if rw_score > scores[i]:
                        population[i] = rw
                        scores[i] = rw_score

                # Update global best
                if scores[i] > best_score:
                    best_build = population[i].copy()
                    best_score = scores[i]

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
                population_size=self.params.pop_size,
            )

        return best_build, best_score

    def _attract(self, dim: np.ndarray, bright: np.ndarray) -> np.ndarray:
        """Move dim firefly toward bright firefly via slot adoption."""
        new_build = dim.copy()
        # Adopt slots from bright firefly with some probability
        mask = np.random.random(size=new_build.shape) < 0.3
        new_build[mask] = bright[mask]
        return new_build

    def _random_walk(self, build: np.ndarray) -> np.ndarray:
        """Random walk: small destroy and repair."""
        n_rem = self.params.n_removal
        partial = random_removal(build, n_rem, self.problem)
        repaired = greedy_insertion(partial, self.budget, self.problem)
        return repaired
