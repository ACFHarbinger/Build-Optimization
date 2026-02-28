"""
Artificial Bee Colony (ABC) algorithm for Build Optimization.

Three agent types — employed, onlooker, and scout bees — cooperate to
explore and exploit the build solution space. Food sources represent
feasible item-slot allocations.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..operators.repair_operators import greedy_insertion
from .params import ABCParams


class ABCSolver(PolicyVizMixin):
    """
    Artificial Bee Colony solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: ABCParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run ABC and return the best solution found.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initialise food sources (employed bees)
        # Each source is an np.ndarray build
        sources = [self.problem.random_solution() for _ in range(self.params.n_sources)]
        scores = [self.problem.evaluate(s) for s in sources]
        trials = [0] * self.params.n_sources

        best_idx = int(np.argmax(scores))
        best_build = sources[best_idx].copy()
        best_score = scores[best_idx]

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            # --- Employed bee phase ---
            for i in range(self.params.n_sources):
                # Select a random peer
                peer_idx = random.choice([x for x in range(self.params.n_sources) if x != i])
                neighbour = self._perturb(sources[i], sources[peer_idx])
                nb_score = self.problem.evaluate(neighbour)

                if nb_score > scores[i]:
                    sources[i] = neighbour
                    scores[i] = nb_score
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- Onlooker bee phase ---
            # Selection probability based on fitness
            min_s = min(scores)
            shifted = [s - min_s + 1e-9 for s in scores]
            total = sum(shifted)
            probs = [s / total for s in shifted]

            for _ in range(self.params.n_sources):
                i = self._roulette(probs)
                peer_idx = random.choice([x for x in range(self.params.n_sources) if x != i])
                neighbour = self._perturb(sources[i], sources[peer_idx])
                nb_score = self.problem.evaluate(neighbour)

                if nb_score > scores[i]:
                    sources[i] = neighbour
                    scores[i] = nb_score
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Update global best
            for i in range(self.params.n_sources):
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_build = sources[i].copy()

            # --- Scout bee phase ---
            for i in range(self.params.n_sources):
                if trials[i] > self.params.limit:
                    sources[i] = self.problem.random_solution()
                    scores[i] = self.problem.evaluate(sources[i])
                    trials[i] = 0

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
                n_sources=self.params.n_sources,
            )

        return best_build, best_score

    def _perturb(self, current: np.ndarray, peer: np.ndarray) -> np.ndarray:
        """
        Perturb the current solution using information from a peer.
        Mimics ABC's interpolation by picking some slots from peer and repairing.
        """
        new_build = current.copy()

        # Pick some random slots to 're-evaluate' using peer's items or random
        n = self.params.n_removal
        slots_to_replace = random.sample(range(self.problem.num_slots), min(n, self.problem.num_slots))

        # For these slots, try peer's items or just clear them
        for slot in slots_to_replace:
            if random.random() < 0.5:
                new_build[slot] = peer[slot]
            else:
                new_build[slot] = -1

        # Ensure feasibility after peer injection, then repair empty slots
        # Actually greedy_insertion handles budget
        repaired = greedy_insertion(new_build, self.budget, self.problem)

        if not self.problem.is_feasible(repaired):
            # If still infeasible (shouldn't happen with greedy_insertion), return current
            return current

        return repaired

    @staticmethod
    def _roulette(probs: List[float]) -> int:
        """Roulette-wheel selection."""
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1
