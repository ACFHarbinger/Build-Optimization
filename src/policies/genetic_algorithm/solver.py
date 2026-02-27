"""
Genetic Algorithm (GA) for VRPP.

Population of route solutions evolved via tournament selection, order
crossover (OX), and random relocate mutation.  Elitism preserves the
best individual across generations.

Reference:
    Holland, "Adaptation in Natural and Artificial Systems", 1975.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.repair_operators import greedy_insertion
from .params import GAParams


class GASolver(PolicyVizMixin):
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

        start = time.time()

        # Initialise population
        population = self._init_population()
        fitnesses = [self._evaluate(ind) for ind in population]

        best_idx = int(np.argmax(fitnesses))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = fitnesses[best_idx]

        for gen in range(self.params.max_generations):
            if time.time() - start > self.params.time_limit:
                break

            new_population: List[List[List[int]]] = []

            # Elitism: carry forward the best
            new_population.append(copy.deepcopy(best_routes))

            while len(new_population) < self.params.pop_size:
                # Tournament selection
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)

                # Crossover
                child = self._crossover(p1, p2) if random.random() < self.params.crossover_rate else copy.deepcopy(p1)

                # Mutation
                if random.random() < self.params.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population
            fitnesses = [self._evaluate(ind) for ind in population]

            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_profit:
                best_routes = copy.deepcopy(population[gen_best_idx])
                best_profit = fitnesses[gen_best_idx]

            self._viz_record(
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
        """Initialise population with randomised NN solutions."""
        from logic.src.policies.operators.heuristics.initialization import build_nn_routes

        population = []
        for _ in range(self.params.pop_size):
            # Shuffle node order for diversity
            nodes_shuffled = self.nodes[:]
            random.shuffle(nodes_shuffled)
            routes = build_nn_routes(
                nodes=nodes_shuffled,
                mandatory_nodes=self.mandatory_nodes,
                wastes=self.wastes,
                capacity=self.capacity,
                dist_matrix=self.dist_matrix,
                R=self.R,
                C=self.C,
            )
            population.append(routes)
        return population

    def _tournament_select(
        self,
        population: List[List[List[int]]],
        fitnesses: List[float],
    ) -> List[List[int]]:
        """Select individual via tournament selection."""
        indices = random.sample(
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

        if len(p2_flat) < 2:
            return copy.deepcopy(parent1)

        a = random.randint(0, len(p2_flat) - 1)
        b = random.randint(a, min(a + max(1, len(p2_flat) // 3), len(p2_flat)))
        segment = p2_flat[a:b]
        segment_set = set(segment)

        remaining = [n for n in p1_flat if n not in segment_set]
        insert_pos = min(a, len(remaining))
        child_flat = remaining[:insert_pos] + segment + remaining[insert_pos:]

        # Rebuild routes respecting capacity
        child_routes: List[List[int]] = []
        curr_route: List[int] = []
        load = 0.0
        for node in child_flat:
            waste = self.wastes.get(node, 0.0)
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    child_routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            child_routes.append(curr_route)

        # Ensure mandatory nodes are present
        visited = {n for r in child_routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                child_routes.append([n])

        return child_routes

    def _mutate(self, routes: List[List[int]]) -> List[List[int]]:
        """Random relocate mutation."""
        flat = [n for r in routes for n in r]
        if not flat:
            return routes
        node = random.choice(flat)
        new_routes = [[n for n in r if n != node] for r in routes]
        new_routes = [r for r in new_routes if r]
        with contextlib.suppress(Exception):
            new_routes = greedy_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                mandatory_nodes=self.mandatory_nodes,
            )
        return new_routes

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
