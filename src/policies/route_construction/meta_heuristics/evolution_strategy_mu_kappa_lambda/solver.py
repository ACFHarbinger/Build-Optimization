"""
(μ,κ,λ) Evolution Strategy for Vehicle Routing Problems (VRP).

This module implements a structurally correct Evolution Strategy (ES) specifically
adapted for discrete routing domains. It follows the metric-based design guidelines
and the age-based selection scheme described in Emmerich et al. (2015).

Key Algorithmic Components:
    - **Self-Adaptive Mutation**: Each individual evolves its own mutation strength
      (n_removal), serving as the discrete analog to the continuous step-size σ
      .
    - **Independent Recombination Selection**: Parents are sampled with replacement
      to maintain high selection pressure.
    - **(μ,κ,λ) Selection Pool**: The survivor pool consists of λ offspring and
      μ parents whose age has not exceeded κ generations.
    - **Memetic Refinement**: Post-mutation local search ensures offspring reach
      local minima within the discrete landscape.

References:
    [2] Emmerich, M., Shir, O. M., & Wang, H. (2015). Evolution Strategies.
        In: Handbook of Natural Computing.
"""

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
from logic.src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.individual import (
    Individual,
)
from logic.src.policies.route_construction.meta_heuristics.evolution_strategy_mu_kappa_lambda.params import (
    MuKappaLambdaESParams,
)


class MuKappaLambdaESSolver:
    """
    (μ,κ,λ) Evolution Strategy solver adapted for VRP.

    This solver optimizes the Vehicle Routing Problem with Profits (VRPP) by
    simulating a generational evolutionary process. It employs deterministic
    truncation selection over an age-limited pool.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MuKappaLambdaESParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initializes the solver and pre-instantiates local search operators.

        Args:
            dist_matrix: Distance matrix where [0] is the depot.
            wastes: Mapping of node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste.
            C: Cost per unit distance.
            params: Configuration dataclass for (μ,κ,λ) parameters.
            mandatory_nodes: Nodes that must be included in feasible solutions.
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
        self.np_rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()

        # Pre-instantiate local search for memetic refinement
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

        self.n_evaluations = 0
        self.best_individual: Optional[Individual] = None
        self.convergence_curve: List[float] = []

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Executes the optimization loop according to the (μ,κ,λ)-ES scheme.

        This follows the high-level loop defined in Algorithm 1.

        Returns:
            A tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # Step 1: Initialize population P_0 with μ parents
        parents = self._initialize_population()
        for ind in parents:
            ind.fitness = self._evaluate(ind.routes)

        self._update_best(parents)

        # Evolution loop
        generation = 0
        while generation < self.params.max_iterations:
            if self.params.time_limit > 0 and (time.process_time() - start_time) > self.params.time_limit:
                break

            # Variation: Generate λ offspring via recombination and self-adaptive mutation
            offspring = []
            for _ in range(self.params.lambda_):
                selected_parents = self._select_parents_for_recombination(parents)
                child = self._recombine(selected_parents)
                mutated_child = self._mutate(child)
                mutated_child.fitness = self._evaluate(mutated_child.routes)
                offspring.append(mutated_child)

            # Selection: Filter parents by age κ and combine with offspring
            eligible_parents = [p for p in parents if p.age <= self.params.kappa]
            selection_pool = eligible_parents + offspring

            # Deterministic truncation: select best μ individuals
            selection_pool.sort(key=lambda ind: ind.fitness, reverse=True)
            parents = selection_pool[: self.params.mu]

            # Increment age of surviving individuals
            for p in parents:
                if p in eligible_parents:
                    p.age += 1

            self._update_best(parents)
            self.convergence_curve.append(self.best_individual.fitness)  # type: ignore[union-attr]

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=generation,
                best_profit=self.best_individual.fitness,  # type: ignore[union-attr]
                best_cost=self._cost(self.best_individual.routes),  # type: ignore[union-attr]
                population_size=self.params.mu,
            )

            generation += 1

        if self.best_individual:
            return (
                self.best_individual.routes,
                self.best_individual.fitness,
                self._cost(self.best_individual.routes),
            )
        return [], 0.0, 0.0

    def _initialize_population(self) -> List[Individual]:
        """
        Creates the initial parent population P_0.

        Initial solutions are built using a randomized nearest-neighbor
        heuristic to ensure initial diversity.

        Returns:
            A list of μ individuals initialized with age 1.
        """
        population = []
        for _ in range(self.params.mu):
            routes = build_greedy_routes(
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.rng,
            )
            ind = Individual(routes=routes, age=1, mutation_strength=float(self.params.n_removal))
            population.append(ind)
        return population

    def _select_parents_for_recombination(self, parents: List[Individual]) -> List[Individual]:
        """
        Uniformly samples ρ parents for recombination.

        Samples are drawn independently and WITH replacement, as specified
        for canonical Evolution Strategies.

        Args:
            parents: Current parent population.

        Returns:
            List of ρ selected individuals.
        """
        return self.rng.choices(parents, k=self.params.rho)

    def _recombine(self, parents: List[Individual]) -> Individual:
        """
        Creates an offspring by combining structural data from ρ parents.

        Implements Discrete Recombination for routes and Intermediate
        Recombination for the mutation strength parameter.

        Args:
            parents: List of ρ parent individuals.

        Returns:
            A new offspring individual with age 1.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if self.params.recombination_type == "intermediate":
            # Intermediate Recombination: merge all nodes and reconstruct
            all_nodes = set()
            for parent in parents:
                for route in parent.routes:
                    all_nodes.update(route)
            routes = []
            if all_nodes:
                if use_profit:
                    routes = greedy_profit_insertion(
                        routes=[],
                        removed_nodes=sorted(list(all_nodes)),
                        dist_matrix=self.dist_matrix,
                        wastes=self.wastes,
                        capacity=self.capacity,
                        R=self.R,
                        C=self.C,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                    )
                else:
                    routes = greedy_insertion(
                        routes=[],
                        removed_nodes=sorted(list(all_nodes)),
                        dist_matrix=self.dist_matrix,
                        wastes=self.wastes,
                        capacity=self.capacity,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=expand_pool,
                    )
        else:
            # Discrete Recombination: pick routes randomly from the parent pool
            all_routes = [route for parent in parents for route in parent.routes]
            routes = []
            if all_routes:
                n_routes = min(len(all_routes), max(1, self.params.mu // 2))
                selected_routes = self.rng.sample(all_routes, n_routes)
                visited: set[int] = set()
                for route in selected_routes:
                    route_set = set(route)
                    if not route_set.intersection(visited):
                        routes.append(route)
                        visited.update(route_set)

        # Average the mutation strength (Intermediate Recombination of strategy params)
        avg_ms = sum(p.mutation_strength for p in parents) / max(1, len(parents))
        return Individual(routes=routes, age=1, mutation_strength=avg_ms)

    def _mutate(self, individual: Individual) -> Individual:
        """
        Performs mutative self-adaptation and discrete perturbation.

        Applies log-normal mutation to the mutation_strength (σ) before
        applying the destroy-repair operator to the routes.

        Args:
            individual: Offspring to mutate.

        Returns:
            The mutated individual.
        """
        mutant = individual.copy()
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        # Step 1: Self-Adapt Strategy Parameter (Eq. 5)
        n_global = self.np_rng.standard_normal()
        n_local = self.np_rng.standard_normal()

        mutant.mutation_strength *= np.exp(self.params.tau_global * n_global + self.params.tau_local * n_local)
        # Boundary control for strategy parameter
        mutant.mutation_strength = np.clip(mutant.mutation_strength, 1.0, 10.0)

        n_remove = int(round(mutant.mutation_strength))

        # Step 2: Object Variable Mutation (Destroy-Repair)
        if mutant.routes:
            try:
                partial, removed = random_removal(mutant.routes, n_remove, rng=self.rng)
                if removed:
                    if use_profit:
                        mutant.routes = greedy_profit_insertion(
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
                        mutant.routes = greedy_insertion(
                            partial,
                            removed,
                            self.dist_matrix,
                            self.wastes,
                            self.capacity,
                            mandatory_nodes=self.mandatory_nodes,
                            expand_pool=expand_pool,
                        )
                # Memetic optimization
                mutant.routes = self.ls.optimize(mutant.routes)
            except Exception:
                pass
        return mutant

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Calculates net profit and increments evaluation counter."""
        self.n_evaluations += 1
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates total distance traveled across all routes."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total

    def _update_best(self, population: List[Individual]):
        """Updates the global elitist solution tracker."""
        current_best = max(population, key=lambda ind: ind.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
