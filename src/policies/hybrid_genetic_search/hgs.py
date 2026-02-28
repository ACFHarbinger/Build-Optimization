"""
Hybrid Genetic Search (HGS) Solver for Build Optimization.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from .evolution import discrete_crossover, evaluate, update_biased_fitness
from .individual import Individual
from .params import HGSParams


class HGSSolver(PolicyVizMixin):
    """
    Hybrid Genetic Search solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: HGSParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the Hybrid Genetic Search algorithm.

        Returns:
            Tuple[np.ndarray, float]: Best build and its score.
        """
        start_time = time.time()

        # 1. Initial Population
        population: List[Individual] = []
        for _ in range(self.params.population_size):
            b = self.problem.random_solution()
            ind = Individual(b)
            evaluate(ind, self.problem)
            population.append(ind)

        update_biased_fitness(population, self.params.elite_size)
        best_score = max(ind.score for ind in population)
        best_build = max(population, key=lambda x: x.score).build.copy()

        iteration = 0
        while time.time() - start_time < self.params.time_limit:
            iteration += 1

            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            child = discrete_crossover(p1, p2)

            # 3. Education (Simple Local Search)
            # In HGS, education is key. We can use a simple neighborhood search.
            child_build = child.build.copy()
            # Try a few random swaps
            for _ in range(5):
                idx1, idx2 = random.sample(range(self.problem.num_slots), 2)
                child_build[idx1], child_build[idx2] = child_build[idx2], child_build[idx1]

            if self.problem.is_feasible(child_build):
                child.build = child_build

            evaluate(child, self.problem)
            population.append(child)

            if child.score > best_score:
                best_score = child.score
                best_build = child.build.copy()

            # 4. Survivor Selection
            if len(population) > self.params.population_size + self.params.n_offspring:
                update_biased_fitness(population, self.params.elite_size)
                # Sort by fitness asc (lower is better)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                child_profit=child.score,
                population_size=len(population),
            )

        return best_build, best_score

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Binary tournament selection based on biased fitness."""

        def tournament():
            i1, i2 = random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()
