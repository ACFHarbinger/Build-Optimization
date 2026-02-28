"""
League Championship Algorithm (LCA) for Build Optimization.

Teams play matches; losers learn from winners by adopting parts of their builds.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import random_removal
from ..operators.repair_operators import greedy_insertion
from .params import LCAParams


class LCASolver(PolicyVizMixin):
    """
    League Championship Algorithm solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: LCAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run LCA and return the best build found.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initialise teams (builds)
        teams = [self.problem.random_solution() for _ in range(self.params.n_teams)]
        scores = [self.problem.evaluate(t) for t in teams]

        best_idx = int(np.argmax(scores))
        best_build = teams[best_idx].copy()
        best_score = scores[best_idx]

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # Weekly matches
            order = list(range(self.params.n_teams))
            random.shuffle(order)

            for k in range(0, len(order) - 1, 2):
                a_i, b_i = order[k], order[k + 1]

                sa, sb = scores[a_i], scores[b_i]

                # Determine winner/loser
                if sa > sb:
                    winner_i, loser_i = a_i, b_i
                elif sb > sa:
                    winner_i, loser_i = b_i, a_i
                else:
                    winner_i, loser_i = (a_i, b_i) if random.random() < 0.5 else (b_i, a_i)

                # Loser generates new formation
                if random.random() < self.params.crossover_prob:
                    new_build = self._crossover(teams[loser_i], teams[winner_i])
                else:
                    new_build = self._perturb(teams[loser_i])

                new_score = self.problem.evaluate(new_build)
                teams[loser_i] = new_build
                scores[loser_i] = new_score

                if new_score > best_score:
                    best_build = new_build.copy()
                    best_score = new_score

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                n_teams=self.params.n_teams,
            )

        return best_build, best_score

    def _perturb(self, build: np.ndarray) -> np.ndarray:
        """Small destroy and repair."""
        partial = random_removal(build, self.params.n_removal, self.problem)
        return greedy_insertion(partial, self.budget, self.problem)

    def _crossover(self, loser: np.ndarray, winner: np.ndarray) -> np.ndarray:
        """Adopt some slots from the winner."""
        child = loser.copy()
        mask = np.random.random(size=child.shape) < 0.4
        child[mask] = winner[mask]

        if not self.problem.is_feasible(child):
            partial = random_removal(child, self.params.n_removal, self.problem)
            child = greedy_insertion(partial, self.budget, self.problem)
        return child
