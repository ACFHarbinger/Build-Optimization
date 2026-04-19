"""
HGS-ALNS Hybrid Solver.

This module implements a hybrid approach where Hybrid Genetic Search (HGS)
uses Adaptive Large Neighborhood Search (ALNS) for its education phase.

Attributes:
    None

Example:
    >>> from logic.src.policies.hybrid_genetic_search_with_adaptive_large_neighborhood_search import HGSALNSSolver
    >>> result = solver.solve()
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.crossover_recombination import ordered_crossover
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns import (
    ALNSSolver,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search import Individual
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution import (
    evaluate,
    update_biased_fitness,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.params import (
    HGSALNSParams,
)


class HGSALNSSolver(HGSSolver):
    """
    Hybrid solver that combines HGS with ALNS as a local search optimizer.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSALNSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the hybrid HGS-ALNS solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: HGSALNSParams containing HGS and ALNS configurations.
            mandatory_nodes: Optional list of mandatory node indices.
        """
        # Initialize parent HGSSolver with HGS params
        super().__init__(dist_matrix, wastes, capacity, R, C, params.hgs_params, mandatory_nodes)

        self.hgs_alns_params = params

        # Initialize ALNS solver for education phase
        self.alns_solver = ALNSSolver(dist_matrix, wastes, capacity, R, C, params.alns_params)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm with ALNS-based education.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        population: List[Individual] = []

        # 1. Initial Population
        for _ in range(self.hgs_alns_params.hgs_params.mu):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt, expand_pool=self.params.vrpp)
            evaluate(ind, self.split_manager)
            population.append(ind)

        update_biased_fitness(
            population,
            self.hgs_alns_params.hgs_params.nb_elite,
            self.hgs_alns_params.hgs_params.nb_granular,
        )

        start_time = time.process_time()
        it = 0
        last_improvement_it = 0
        best_profit_so_far = max(ind.profit_score for ind in population)
        while last_improvement_it < self.hgs_alns_params.hgs_params.n_iterations_no_improvement:
            if (
                self.hgs_alns_params.time_limit > 0
                and time.process_time() - start_time >= self.hgs_alns_params.time_limit
            ):
                break
            it += 1
            # 2. Selection & Crossover
            p1, p2 = self._select_parents(population)
            child = ordered_crossover(p1, p2)

            # 3. Hybrid Education (Mutation with ALNS)
            if self.random.random() < self.hgs_alns_params.hgs_params.mutation_rate:
                # Need to have routes assigned before ALNS
                evaluate(child, self.split_manager)
                if child.routes:
                    # Convert child to ALNS format and run ALNS
                    # ALNSSolver.solve accepts List[List[int]] as initial_solution
                    new_routes, profit, cost = self.alns_solver.solve(initial_solution=child.routes)

                    # Update individual with ALNS results
                    child.routes = new_routes
                    child.profit_score = profit
                    child.cost = cost

                    # Reconstruct giant tour from ALNS routes to maintain HGS consistency
                    new_gt = []
                    visited_nodes = set()
                    for r in new_routes:
                        new_gt.extend(r)
                        visited_nodes.update(r)

                    unvisited = [n for n in self.nodes if n not in visited_nodes]
                    new_gt.extend(unvisited)
                    child.giant_tour = new_gt
                else:
                    # Fallback to standard local search if split failed to produce routes
                    child = self.ls.optimize(child)
                    evaluate(child, self.split_manager)
            else:
                evaluate(child, self.split_manager)
            child.expand_pool = self.params.vrpp
            population.append(child)

            if child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                last_improvement_it = 0
            else:
                last_improvement_it += 1

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(population),
            )

            # 4. Survivor Selection
            if len(population) > self.hgs_alns_params.hgs_params.mu * 2:
                update_biased_fitness(
                    population,
                    self.hgs_alns_params.hgs_params.nb_elite,
                    self.hgs_alns_params.hgs_params.nb_granular,
                )
                population.sort(key=lambda x: x.fitness)
                population = population[: self.hgs_alns_params.hgs_params.mu]

        update_biased_fitness(
            population,
            self.hgs_alns_params.hgs_params.nb_elite,
            self.hgs_alns_params.hgs_params.nb_granular,
        )
        best_ind = min(population, key=lambda x: -x.profit_score)

        return best_ind.routes, best_ind.profit_score, best_ind.cost
