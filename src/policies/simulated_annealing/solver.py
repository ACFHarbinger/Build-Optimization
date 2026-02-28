"""
Simulated Annealing (SA) for Build Optimization.

Classic meta-heuristic drawing analogy from metallurgical annealing.
Non-improving moves are accepted with Boltzmann probability
exp(Δscore / T), where T is a temperature parameter that decays
geometrically via T *= alpha.
"""

import math
import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .params import SAParams


class SASolver(PolicyVizMixin):
    """
    Simulated Annealing solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: SAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        # Use standard build operators: random removal and greedy insertion
        from ..operators.destroy_operators import random_removal
        from ..operators.repair_operators import greedy_insertion

        self.destroy_op = random_removal
        self.repair_op = greedy_insertion

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run simulated annealing.

        Returns:
            Tuple of (build, score).
        """
        start = time.time()

        # Initial solution
        current_build = self.problem.greedy_solution()
        current_score = self.problem.evaluate(current_build)

        best_build = current_build.copy()
        best_score = current_score

        T = self.params.initial_temp

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Perturb: small destroy and repair
            try:
                # Remove n_removal items
                partial_build = self.destroy_op(current_build, self.params.n_removal, self.problem)

                # Repair
                new_build = self.repair_op(partial_build, self.budget, self.problem)

                if not self.problem.is_feasible(new_build):
                    continue

                new_score = self.problem.evaluate(new_build)
            except Exception:
                continue

            delta = new_score - current_score

            # Boltzmann acceptance (for maximization)
            if delta >= 0:
                accept = True
            elif T > 1e-10:
                accept = random.random() < math.exp(delta / T)
            else:
                accept = False

            if accept:
                current_build = new_build
                current_score = new_score

                if current_score > best_score:
                    best_build = current_build.copy()
                    best_score = current_score

            # Geometric cooling
            T = max(self.params.min_temp, T * self.params.alpha)

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,  # viz record legacy names
                best_cost=-best_score,
                temperature=T,
            )

        return best_build, best_score
