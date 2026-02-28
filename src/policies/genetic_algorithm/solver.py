"""
Genetic Algorithm (GA) for Build Optimization.

Population of build solutions evolved via tournament selection,
uniform crossover, and point mutation. Elitism preserves the
best individual across generations.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .params import GAParams


class GASolver(PolicyVizMixin):
    """
    Genetic Algorithm solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: GAParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run GA optimisation.

        Returns:
            Tuple of (best_build, best_score).
        """
        start = time.time()

        # Initialise population
        population = self._init_population()
        fitnesses = [self.problem.evaluate(ind) for ind in population]

        best_idx = int(np.argmax(fitnesses))
        best_build = population[best_idx].copy()
        best_score = fitnesses[best_idx]

        for gen in range(self.params.max_generations):
            if time.time() - start > self.params.time_limit:
                break

            new_population: List[np.ndarray] = []

            # Elitism: carry forward the best
            new_population.append(best_build.copy())

            while len(new_population) < self.params.pop_size:
                # Tournament selection
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                # Crossover
                child = self._crossover(p1, p2) if random.random() < self.params.crossover_rate else p1.copy()

                # Mutation
                if random.random() < self.params.mutation_rate:
                    child = self._mutate(child)

                # Ensure feasibility (naive)
                if not self.problem.is_feasible(child):
                    # If infeasible, try to repair or just re-randomize
                    child = self.problem.random_solution()

                new_population.append(child)

            population = new_population
            fitnesses = [self.problem.evaluate(ind) for ind in population]

            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_score:
                best_score = fitnesses[gen_best_idx]
                best_build = population[gen_best_idx].copy()

            self._viz_record(
                iteration=gen,
                best_profit=best_score,  # legacy name
                best_cost=-best_score,
                pop_size=len(population),
            )

        return best_build, best_score

    def _init_population(self) -> List[np.ndarray]:
        """Initialise population with random feasible builds."""
        return [self.problem.random_solution() for _ in range(self.params.pop_size)]

    def _tournament_select(
        self,
        population: List[np.ndarray],
        fitnesses: List[float],
    ) -> np.ndarray:
        """Select individual via tournament selection."""
        indices = random.sample(
            range(len(population)),
            min(self.params.tournament_size, len(population)),
        )
        best = max(indices, key=lambda i: fitnesses[i])
        return population[best]

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> np.ndarray:
        """Uniform crossover: pick slot from either parent."""
        child = parent1.copy()
        mask = np.random.random(size=child.shape) < 0.5
        child[mask] = parent2[mask]
        return child

    def _mutate(self, build: np.ndarray) -> np.ndarray:
        """Point mutation: pick a random slot and pick a new item."""
        new_build = build.copy()
        slot_idx = random.randint(0, self.problem.num_slots - 1)

        # Pick a random item for this slot (simple)
        item_indices = np.where(self.problem.slot_ids == slot_idx)[0]
        if len(item_indices) > 0:
            new_build[slot_idx] = random.choice(item_indices)

        return new_build
