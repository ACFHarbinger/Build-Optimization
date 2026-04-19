"""
(μ+λ) Evolution Strategy for VRPP.

This module implements a rigorous, elitist (μ+λ) Evolution Strategy.
The search in Evolution Strategies is characterized by the alternating
application of variation and selection operators.

Algorithm:
    1. Initialize a parent population of μ solutions.
    2. For each generation:
        a. Variation: Generate λ offspring by recombining parents and
           applying mutative perturbations.
        b. Evaluation: Calculate the net profit for all λ offspring.
        c. Selection: Combine the μ parents and λ offspring into a single
           pool, sort by fitness, and select the top μ individuals to survive
           into the next generation.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import (
    build_greedy_routes,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import (
    KSACOParams,
)
from logic.src.policies.route_construction.meta_heuristics.evolution_strategy_mu_plus_lambda.params import (
    MuPlusLambdaESParams,
)


class MuPlusLambdaESSolver:
    """
    (μ+λ) Evolution Strategy solver for the Vehicle Routing Problem with Profits.

    This solver enforces strict elitism by allowing parent solutions to compete
    directly with their offspring for survival into the next generation.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MuPlusLambdaESParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        r"""
        Initializes the (μ+λ)-ES solver and pre-allocates the local search operator.

        Args:
            dist_matrix: Distance matrix of shape (n+1, n+1), where index 0 is the depot.
            wastes: Mapping of node IDs to their available waste quantities.
            capacity: Maximum capacity constraint for a single vehicle route.
            R: Revenue generated per unit of waste collected.
            C: Cost incurred per unit of distance traveled.
            params: Configuration dataclass defining $\mu$, $\lambda$, and limits.
            mandatory_nodes: List of node IDs that must be included in any feasible solution.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.rng = random.Random(params.seed) if params.seed is not None else random.Random()

        # Pre-instantiate local search for reuse to prevent instantiation overhead
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            vrpp=self.params.vrpp,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=self.params.seed,
        )
        self.ls = ACOLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Executes the (μ+λ) Evolution Strategy optimization loop.

        Returns:
            A tuple containing:
                - best_routes: The routing sequence of the best solution found.
                - best_profit: The net profit (fitness) of the best solution.
                - best_cost: The total routing distance of the best solution.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialize parent population (μ)
        parents = [self._initialize_solution() for _ in range(self.params.mu)]
        parent_fitnesses = [self._evaluate(p) for p in parents]

        best_routes = []
        best_profit = -float("inf")
        best_cost = 0.0

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            offspring_population = []
            offspring_fitnesses = []

            # --- Variation: Generate λ offspring ---
            for _ in range(self.params.lambda_):
                # Uniform random selection of 2 parents for recombination
                p1, p2 = self.rng.sample(parents, 2)

                offspring = self._recombine_and_mutate(p1, p2)
                offspring_fitness = self._evaluate(offspring)

                offspring_population.append(offspring)
                offspring_fitnesses.append(offspring_fitness)

            # --- Selection: Elitist (μ+λ) ---
            # Combine the current μ parents and λ offspring into a single pool
            combined_population = parents + offspring_population
            combined_fitness = parent_fitnesses + offspring_fitnesses

            # Sort combined descending by fitness
            sorted_indices = np.argsort(combined_fitness)[::-1]

            # Select the top μ individuals to survive to the next generation
            survivor_indices = sorted_indices[: self.params.mu]
            parents = [copy.deepcopy(combined_population[i]) for i in survivor_indices]
            parent_fitnesses = [combined_fitness[i] for i in survivor_indices]

            # Update global best tracking
            current_best_fit = parent_fitnesses[0]
            if current_best_fit > best_profit:
                best_profit = current_best_fit
                best_routes = copy.deepcopy(parents[0])
                best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.mu,
            )

        return best_routes, best_profit, best_cost

    def _initialize_solution(self) -> List[List[int]]:
        """
        Initializes a single solution using the greedy profit-aware heuristic.
        """
        return build_greedy_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.rng,
        )

    def _recombine_and_mutate(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """
        Applies discrete recombination and mutation to generate a single offspring.
        """
        if not parent1 or not parent2:
            return copy.deepcopy(parent1)

        n_remove = max(3, self.params.n_removal)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        p2_nodes = [node for route in parent2 for node in route]
        if not p2_nodes:
            return copy.deepcopy(parent1)

        # Crossover/Recombination
        selected_p2_nodes = self.rng.sample(p2_nodes, min(n_remove, len(p2_nodes)))

        offspring = copy.deepcopy(parent1)
        for route in offspring:
            for node in selected_p2_nodes:
                if node in route:
                    route.remove(node)
        offspring = [route for route in offspring if route]

        # Mutation: Random destruction and greedy repair
        try:
            partial_routes, additional_removed = random_removal(offspring, n_remove, rng=self.rng)
            nodes_to_insert = sorted(list(set(selected_p2_nodes + additional_removed)))
            if use_profit:
                repaired_routes = greedy_profit_insertion(
                    partial_routes,
                    nodes_to_insert,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            else:
                repaired_routes = greedy_insertion(
                    partial_routes,
                    nodes_to_insert,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            return self.ls.optimize(repaired_routes)
        except Exception:
            return copy.deepcopy(parent1)

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Evaluates the fitness (net profit) of a given solution."""
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates the total routing distance for a given solution."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
