"""
Genetic Algorithm (GA) for VRPP.

Population of route solutions evolved via tournament selection, order
crossover (OX), and random relocate mutation.  Elitism preserves the
best individual across generations.

References:
    Holland, J. H. "Adaptation in Natural and Artificial Systems", 1975.
    Prins, C. "A simple and effective evolutionary
    algorithm for the vehicle routing problem", 2004.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.crossover_recombination import ordered_crossover
from logic.src.policies.helpers.operators.destroy_ruin.random import random_removal
from logic.src.policies.helpers.operators.recreate_repair.greedy import (
    greedy_insertion,
    greedy_profit_insertion,
)
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import (
    build_greedy_routes,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual

from .params import GAParams


class GASolver:
    """
    Genetic Algorithm solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GAParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run GA optimisation.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialise population
        population = self._init_population()
        fitnesses = [self._evaluate(ind) for ind in population]

        best_idx = int(np.argmax(fitnesses))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = fitnesses[best_idx]

        for gen in range(self.params.max_generations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            new_population: List[List[List[int]]] = []

            # Prins (2004) Style Elitism: carry forward the best individual
            new_population.append(copy.deepcopy(best_routes))

            while len(new_population) < self.params.pop_size:
                # Tournament selection
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                # OX Crossover (standard for VRP)
                if self.random.random() < self.params.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = copy.deepcopy(p1)

                # Mutation (Inversion/Relocate)
                if self.random.random() < self.params.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population
            fitnesses = [self._evaluate(ind) for ind in population]

            # Update global best
            gen_best_idx = int(np.argmax(fitnesses))

            # Prins (2004): Local search improvement on the best individual of the generation
            # This ensures the population "drifts" towards local optima
            population[gen_best_idx] = self._local_search(population[gen_best_idx])
            fitnesses[gen_best_idx] = self._evaluate(population[gen_best_idx])

            if fitnesses[gen_best_idx] > best_profit:
                best_routes = copy.deepcopy(population[gen_best_idx])
                best_profit = fitnesses[gen_best_idx]

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=gen,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
                pop_size=len(population),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _init_population(self) -> List[List[List[int]]]:
        """Initialise population with randomised greedy solutions."""
        population = []
        for _ in range(self.params.pop_size):
            # Use seed/rng from params
            routes = build_greedy_routes(
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
            )
            population.append(routes)
        return population

    def _tournament_select(
        self,
        population: List[List[List[int]]],
        fitnesses: List[float],
    ) -> List[List[int]]:
        """Select individual via tournament selection."""
        indices = self.random.sample(
            range(len(population)),
            min(self.params.tournament_size, len(population)),
        )
        best = max(indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best])

    def _crossover(
        self,
        parent1: List[List[int]],
        parent2: List[List[int]],
    ) -> List[List[int]]:
        """OX crossover: inject a segment from parent2 into parent1."""
        p1_flat = [n for r in parent1 for n in r]
        p2_flat = [n for r in parent2 for n in r]

        if not p1_flat or not p2_flat:
            return copy.deepcopy(parent1)

        ind1 = Individual(p1_flat)
        ind2 = Individual(p2_flat)
        child_ind = ordered_crossover(ind1, ind2, self.random)
        child_flat = child_ind.giant_tour

        # Rebuild routes respecting capacity and profit-awareness
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            return greedy_profit_insertion(
                [],
                child_flat,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            return greedy_insertion(
                [],
                child_flat,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _mutate(self, routes: List[List[int]]) -> List[List[int]]:
        """Random relocate mutation using shared ruin/recreate."""
        if not any(routes):
            return routes

        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        partial, removed = random_removal(routes, 1, self.random)
        try:
            if use_profit:
                return greedy_profit_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                return greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
        except Exception:
            return routes

    def _local_search(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Prins (2004) style local search (2-opt).

        Systematically improves the best individual to a local optimum.
        """
        improved = True
        current_routes = [list(r) for r in routes]
        current_profit = self._evaluate(current_routes)

        while improved:
            improved = False
            # Try 2-opt on each route
            for r_idx in range(len(current_routes)):
                route = current_routes[r_idx]
                if len(route) < 2:
                    continue

                # Full 2-opt on this route
                n = len(route)
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        # Reverse segment
                        new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]

                        # Calculate profit delta without full re-evaluation for efficiency
                        # Delta distance = d(i-1, i) + d(j, j+1) - d(i-1, j) - d(j, i+1)
                        # But since it's only one route, we can just re-evaluate if needed or use a helper
                        test_routes = [list(r) for r in current_routes]
                        test_routes[r_idx] = new_route
                        new_profit = self._evaluate(test_routes)

                        if new_profit > current_profit + 1e-6:
                            current_routes[r_idx] = new_route
                            current_profit = new_profit
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return current_routes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
