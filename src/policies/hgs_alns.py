"""
HGS-ALNS Hybrid Solver for Build Optimization.

Uses ALNS for the 'education' (local search) phase of HGS.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem

from .adaptive_large_neighborhood_search.alns import ALNSSolver
from .adaptive_large_neighborhood_search.params import ALNSParams
from .hybrid_genetic_search.evolution import discrete_crossover, evaluate, update_biased_fitness
from .hybrid_genetic_search.hgs import HGSSolver
from .hybrid_genetic_search.individual import Individual
from .hybrid_genetic_search.params import HGSParams


class HGSALNSSolver(HGSSolver):
    """
    Hybrid solver that combines HGS with ALNS as a local search optimizer.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: HGSParams,
        alns_education_iterations: int = 50,
    ):
        super().__init__(problem, budget, params)
        self.alns_iter = alns_education_iterations

        # Initialize ALNS solver with limited iterations for intensive education
        alns_params = ALNSParams(
            max_iterations=self.alns_iter,
            time_limit=max(1.0, params.time_limit / 10.0),
        )
        self.alns_solver = ALNSSolver(problem, budget, alns_params)

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the Hybrid Genetic Search algorithm with ALNS-based education.

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

            # 3. Hybrid Education (Mutation with ALNS)
            if random.random() < self.params.mutation_rate:
                # Run ALNS for education
                new_build, new_score = self.alns_solver.solve(initial_solution=child.build)
                child.build = new_build
                child.score = new_score
            else:
                evaluate(child, self.problem)

            population.append(child)

            if child.score > best_score:
                best_score = child.score
                best_build = child.build.copy()

            # 4. Survivor Selection
            if len(population) > self.params.population_size + self.params.n_offspring:
                update_biased_fitness(population, self.params.elite_size)
                population.sort(key=lambda x: x.fitness)
                population = population[: self.params.population_size]

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                child_profit=child.score,
                population_size=len(population),
            )

        return best_build, best_score
