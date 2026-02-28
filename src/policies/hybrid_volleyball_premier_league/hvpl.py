"""
Hybrid Volleyball Premier League (HVPL) Solver for Build Optimization.

Combines ACO for construction and global guidance with
ALNS (Coaching) for local improvement, within a population-based framework.
"""

import time
from typing import List, Tuple

import numpy as np

from core.problem import BuildProblem
from tracking.viz_mixin import PolicyVizMixin

from ..adaptive_large_neighborhood_search.alns import ALNSSolver
from ..ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
from .params import HVPLParams


class HVPLSolver(PolicyVizMixin):
    """
    Hybrid Volleyball Premier League solver for Build Optimization.
    """

    def __init__(
        self,
        problem: BuildProblem,
        budget: float,
        params: HVPLParams,
    ):
        self.problem = problem
        self.budget = budget
        self.params = params

        # Initialize ACO components for constructor and pheromones
        self.aco_internal = KSparseACOSolver(
            problem=problem,
            budget=budget,
            params=params.aco_params,
        )
        self.pheromone = self.aco_internal.pheromone
        self.constructor = self.aco_internal.constructor

        # Initialize ALNS solver for the "Coaching" phase
        self.coaching_solver = ALNSSolver(
            problem=problem,
            budget=budget,
            params=params.alns_params,
        )

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run the HVPL algorithm.

        Returns:
            Tuple of (best_build, best_score).
        """
        start_time = time.time()

        # 1. Initialization: Create the initial population (Teams)
        # Each team is (build, score)
        population: List[Tuple[np.ndarray, float]] = []
        for _ in range(self.params.n_teams):
            build = self.constructor.construct()
            score = self.problem.evaluate(build)
            population.append((build, score))

        best_build, best_score = self._get_best(population)

        # 2. League Season Iterations
        for iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            # 3. Coaching Phase: Apply ALNS to each team
            new_population = []
            for build, _score in population:
                # Coaching session (ALNS solve)
                c_build, c_score = self.coaching_solver.solve(initial_solution=build)
                new_population.append((c_build, c_score))

            population = new_population

            # 4. Global Competition: Update best-so-far
            iter_best_build, iter_best_score = self._get_best(population)
            if iter_best_score > best_score:
                best_build = iter_best_build.copy()
                best_score = iter_best_score

            # 5. Pheromone Update: Global guidance
            self._update_pheromones(best_build, best_score)

            self._viz_record(
                iteration=iteration,
                best_profit=best_score,
                best_cost=-best_score,
                iter_best_profit=iter_best_score,
                population_size=len(population),
            )

            # 6. Substitution Phase: Replace weakest teams
            # Sort by score (higher is better)
            population.sort(key=lambda x: x[1], reverse=True)
            n_sub = int(self.params.n_teams * self.params.sub_rate)

            for i in range(self.params.n_teams - n_sub, self.params.n_teams):
                # Replace with a new solution generated with updated pheromones
                s_build = self.constructor.construct()
                s_score = self.problem.evaluate(s_build)
                population[i] = (s_build, s_score)

        return best_build, best_score

    def _get_best(self, population: List[Tuple[np.ndarray, float]]) -> Tuple[np.ndarray, float]:
        """Get the highest-score solution from the population."""
        best_idx = int(np.argmax([p[1] for p in population]))
        return population[best_idx][0].copy(), population[best_idx][1]

    def _update_pheromones(self, best_build: np.ndarray, best_score: float) -> None:
        """ACS style global pheromone update."""
        if best_build is None or best_score <= 0:
            return

        # Evaporate
        self.pheromone.evaporate_all(self.params.aco_params.rho)

        # Deposit
        delta = self.params.aco_params.elitist_weight * best_score
        for slot_idx, item_idx in enumerate(best_build):
            if item_idx != -1:
                self.pheromone.update_edge(slot_idx, item_idx, delta, evaporate=False)
