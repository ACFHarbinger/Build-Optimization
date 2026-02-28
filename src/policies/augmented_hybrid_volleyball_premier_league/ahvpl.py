"""
Augmented Hybrid Volleyball Premier League (AHVPL) Solver for Build Optimization.

Integrates ACO, GA (HGS-inspired), and ALNS into a unified engine.
"""

import random
import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..adaptive_large_neighborhood_search.alns import ALNSSolver
from ..ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
from .params import AHVPLParams


class BuildIndividual:
    """Represents a candidate build in the AHVPL population."""

    def __init__(self, build: np.ndarray, score: float):
        self.build = build
        self.score = score
        self.fitness = 0.0  # Biased fitness (score + diversity)
        self.rank = 0
        self.diversity = 0.0


def update_biased_fitness(population: List[BuildIndividual], elite_size: int):
    """Compute biased fitness based on score ranking and diversity."""
    if not population:
        return

    # Sort by score descending
    population.sort(key=lambda x: x.score, reverse=True)

    # Ranking contribution
    for i, ind in enumerate(population):
        ind.rank = i

    # Diversity contribution (Hamming distance to others)
    # This is expensive for large populations, so we sample
    for ind in population:
        dist_sum = 0.0
        sampled_neighbors = random.sample(population, min(len(population), 5))
        for other in sampled_neighbors:
            if ind is other:
                continue
            # Overlap distance
            dist_sum += np.sum(ind.build != other.build)
        ind.diversity = dist_sum / (len(sampled_neighbors) + 1e-9)

    # Combine (minimize fitness: lower is better)
    # Fitness = rank + (1 - diversity_rank)
    # For simplicity, we just use a weighted sum here
    max_div = max(ind.diversity for ind in population) + 1e-9
    for ind in population:
        ind.fitness = ind.rank - (ind.diversity / max_div)


class AHVPLSolver(PolicyVizMixin):
    """
    Augmented Hybrid Volleyball Premier League solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: AHVPLParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        # Component solvers
        self.aco_solver = KSparseACOSolver(problem, budget, params.aco_params)
        self.alns_solver = ALNSSolver(problem, budget, params.alns_params)

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the Augmented HVPL algorithm.

        Returns:
            Tuple of (best_build, best_score).
        """
        start_time = time.time()

        # 1. Initialization: ACO
        population = self._initialize_population()
        if not population:
            return self.problem.greedy_solution(), 0.0

        best_ind = max(population, key=lambda x: x.score)
        best_build = best_ind.build.copy()
        best_score = best_ind.score

        # 2. Main Loop
        for iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            # A. Select parents and crossover
            update_biased_fitness(population, self.params.hgs_params.elite_size)
            n_crossovers = max(1, int(len(population) * self.params.hgs_params.crossover_rate))

            children = []
            for _ in range(n_crossovers):
                p1, p2 = self._select_parents(population)
                child_build = self._crossover(p1.build, p2.build)
                child_score = self.problem.evaluate(child_build)
                children.append(BuildIndividual(child_build, child_score))

            population.extend(children)

            # B. Coaching: ALNS on a subset of the population (for speed)
            # Or on all if population is small
            for i in range(len(population)):
                if random.random() < 0.2:  # Only coach 20% to save time
                    improved_build, improved_score = self.alns_solver.solve(initial_solution=population[i].build)
                    population[i] = BuildIndividual(improved_build, improved_score)

            # C. Survivor Selection
            update_biased_fitness(population, self.params.hgs_params.elite_size)
            population.sort(key=lambda x: x.fitness)
            population = population[: self.params.n_teams]

            # D. Update global best
            iter_best = max(population, key=lambda x: x.score)
            if iter_best.score > best_score:
                best_build = iter_best.build.copy()
                best_score = iter_best.score

            # E. Pheromone Update (ACO)
            self.aco_solver._global_pheromone_update(best_build, best_score)

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                iter_best_profit=iter_best.score,
                population_size=len(population),
            )

            # F. Substitution
            n_sub = max(1, int(self.params.n_teams * self.params.sub_rate))
            for i in range(len(population) - n_sub, len(population)):
                b = self.aco_solver.constructor.construct()
                s = self.problem.evaluate(b)
                population[i] = BuildIndividual(b, s)

        return best_build, best_score

    def _initialize_population(self) -> List[BuildIndividual]:
        pop = []
        for _ in range(self.params.n_teams):
            b = self.aco_solver.constructor.construct()
            s = self.problem.evaluate(b)
            pop.append(BuildIndividual(b, s))
        return pop

    def _select_parents(self, population: List[BuildIndividual]) -> Tuple[BuildIndividual, BuildIndividual]:
        """Binary tournament selection."""

        def tournament():
            i1, i2 = random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Simple discrete crossover for builds."""
        child = p1.copy()
        mask = np.random.random(size=child.shape) < 0.5
        child[mask] = p2[mask]

        # Ensure feasibility (budget check is in evaluate/is_feasible)
        if not self.problem.is_feasible(child):
            # Just return p1 if crossover failed badly, or a random repair
            return p1.copy()
        return child
