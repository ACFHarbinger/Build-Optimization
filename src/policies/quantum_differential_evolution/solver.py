"""
Quantum-Inspired Differential Evolution (QDE) for Build Optimization.

Represents each candidate as a quantum amplitude vector q ∈ [0,1]^N.
The trial vector is collapsed to a discrete build via ranking and greedy insertion.
"""

import random
import time
from typing import Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .params import QDEParams


class QDESolver(PolicyVizMixin):
    """
    Quantum-Inspired Differential Evolution solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: QDEParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run QDE and return the best build found.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()
        pop_size = self.params.pop_size
        num_slots = self.problem.num_slots

        # Initialise population: amplitude vectors ∈ [0,1]^num_slots
        population = [np.random.uniform(0.0, 1.0, num_slots) for _ in range(pop_size)]
        builds_pop = [self._collapse(amp) for amp in population]
        scores = [self.problem.evaluate(b) for b in builds_pop]

        best_idx = int(np.argmax(scores))
        best_build = builds_pop[best_idx].copy()
        best_score = scores[best_idx]

        for iteration in range(self.params.max_iterations):
            if time.time() - start > self.params.time_limit:
                break

            for i in range(pop_size):
                # --- Mutation ---
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = random.sample(candidates, 3)
                mutant = np.clip(
                    population[r1] + self.params.F * (population[r2] - population[r3]),
                    0.0,
                    1.0,
                )

                # --- Crossover (binomial) ---
                j_rand = random.randint(0, num_slots - 1)
                trial = np.where(
                    (np.random.uniform(0.0, 1.0, num_slots) < self.params.CR) | (np.arange(num_slots) == j_rand),
                    mutant,
                    population[i],
                )

                # --- Collapse → discrete build ---
                trial_build = self._collapse(trial)
                trial_score = self.problem.evaluate(trial_build)

                # --- Greedy selection ---
                if trial_score >= scores[i]:
                    population[i] = trial
                    builds_pop[i] = trial_build
                    scores[i] = trial_score

                    if trial_score > best_score:
                        best_build = trial_build.copy()
                        best_score = trial_score

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
                population_size=pop_size,
            )

        return best_build, best_score

    def _collapse(self, amplitudes: np.ndarray) -> np.ndarray:
        """
        Collapse amplitude vector to a discrete build.
        Slots are processed in order of their amplitudes.
        """
        ranked_slots = np.argsort(amplitudes)[::-1]

        build = np.full(self.problem.num_slots, -1, dtype=int)

        # We use greedy_insertion logic but restricted to the ranked slots?
        # Actually, let's just use the greedy_insertion directly on a partial build
        # where we decide which slots to prioritize.

        # Simplified: just run greedy_insertion. The amplitudes could be used to
        # weight the items in the greedy step, but that's complex.
        # For now, let's just use the amplitudes as a seed for a randomized greedy?
        # Or better: construct build by iterating through ranked slots and picking best item.

        current_cost = 0.0
        for slot in ranked_slots:
            # Pick best item for this slot that fits in budget
            item_indices = np.where(self.problem.slot_ids == slot)[0]
            if len(item_indices) == 0:
                continue

            best_item = -1
            best_val = -1e9

            for item_idx in item_indices:
                cost = self.problem.costs[item_idx]
                if current_cost + cost <= self.budget:
                    # Score = yield / cost (simple greedy proxy)
                    score = self.problem.yields[item_idx] / (cost + 1e-6)
                    if score > best_val:
                        best_val = score
                        best_item = item_idx

            if best_item != -1:
                build[slot] = best_item
                current_cost += self.problem.costs[best_item]

        return build
