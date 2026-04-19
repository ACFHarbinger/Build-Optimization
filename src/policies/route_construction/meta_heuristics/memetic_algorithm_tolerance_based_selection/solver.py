"""
Memetic Algorithm with Tolerance-based Selection (MA-TS) for VRPP.

EXACT COPY of League Championship Algorithm (LCA) with rigorous nomenclature.

TERMINOLOGY MAPPING (LCA → MA-TS):
- "Teams" → Population (candidate solutions)
- "League Schedule" → Round-Robin Pairwise Matching
- "Playing Strength" → Fitness (objective function value)
- "Match Outcome" → Pairwise Comparison with Infeasibility Tolerance
- "Team Formation" → Solution Structure
- "Seasons" → Generations

Algorithm Structure (Kashan, 2013):
    1. Initialize population of N solutions
    2. For each generation:
        a. Round-Robin Schedule: Random pairwise matching
        b. For each match (i vs j):
            - Determine winner using infeasibility tolerance
            - If |fitness_i - fitness_j| ≤ tolerance: random winner (diversity)
            - Otherwise: higher fitness wins
            - Loser generates new solution (crossover or mutation)
            - Loser ALWAYS accepts new solution (no elitism within match)
        c. Update global best

Key Feature: Infeasibility Tolerance
    - Allows solutions with similar fitness to compete randomly
    - Preserves diversity and prevents premature convergence
    - Critical for bridging isolated feasible basins in constrained problems

Reference:
    Kashan, A. H. (2013). "League Championship Algorithm (LCA): An algorithm
    for global optimization inspired by sport championships."
    Applied Soft Computing, 13(5), 2171-2200.

IMPORTANT: This implementation EXACTLY matches the LCA algorithm with only
           terminology changed from sports metaphors to OR terminology.
           Uses identical RNG attribute name (self.random) and algorithm flow.
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import MemeticAlgorithmToleranceBasedSelectionParams


class MemeticAlgorithmToleranceBasedSelectionSolver:
    """
    Memetic Algorithm with Tolerance-based Selection for VRPP.
    EXACT COPY of LCA with rigorous nomenclature.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MemeticAlgorithmToleranceBasedSelectionParams,
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
        self.mandatory_set = set(self.mandatory_nodes)
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random()

        # Initialize Local Search for refinement
        aco_params = KSACOParams(
            local_search_iterations=self.params.local_search_iterations,
            time_limit=self.params.time_limit,
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run MA-TS and return the best feasible solution.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        # Initialise population (routing solutions)
        # LCA: teams = [...]
        # MA-TS: population = [...]
        population: List[List[List[int]]] = [self._build_random_solution() for _ in range(self.params.population_size)]
        profits: List[float] = [self._evaluate(sol) for sol in population]

        best_idx = int(np.argmax(profits))
        best_routes = copy.deepcopy(population[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Random round-robin schedule for this generation
            # LCA: order = [...]
            # MA-TS: order = [...] (same name for exact match)
            order = list(range(self.params.population_size))
            self.random.shuffle(order)

            for k in range(0, len(order) - 1, 2):
                a_idx = order[k]
                b_idx = order[k + 1]

                pa = profits[a_idx]
                pb = profits[b_idx]

                # Determine winner/loser with infeasibility tolerance
                # A solution within `tolerance_pct` of its opponent is allowed
                # to win based on diversity (random coin flip)
                delta = abs(pa - pb)
                tolerance = self.params.tolerance_pct * (abs(pa) + abs(pb) + 1e-9) / 2.0

                if delta <= tolerance:
                    # Close match — random winner
                    winner, loser = (a_idx, b_idx) if self.random.random() < 0.5 else (b_idx, a_idx)
                elif pa >= pb:
                    winner, loser = a_idx, b_idx
                else:
                    winner, loser = b_idx, a_idx

                # Loser generates new solution
                # LCA: crossover_prob
                # MA-TS: recombination_rate (aliased to crossover_prob in params)
                if self.random.random() < self.params.recombination_rate:
                    new_solution = self._crossover(population[loser], population[winner])
                else:
                    new_solution = self._perturb(population[loser])

                new_profit = self._evaluate(new_solution)

                # Accept new solution (MA-TS always updates the loser)
                population[loser] = new_solution
                profits[loser] = new_profit

                # Track best feasible solution
                if new_profit > best_profit:
                    best_routes = copy.deepcopy(new_solution)
                    best_profit = new_profit
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_teams=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Random node ordering causes different capacity cutoffs, creating
        genuinely diverse initial solutions. Uses self.C for the profitability
        check so that economics are consistent with the solver's _evaluate().
        """
        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return optimized_routes

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """Mutation operator using destroy-repair."""
        try:
            partial, removed = random_removal(routes, self.params.n_removal, self.random)
            if self.params.profit_aware_operators:
                repaired = greedy_profit_insertion(
                    routes=partial,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    C=self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                repaired = greedy_insertion(
                    routes=partial,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            return self.ls.optimize(repaired)
        except Exception:
            return routes

    def _crossover(self, loser_routes: List[List[int]], winner_routes: List[List[int]]) -> List[List[int]]:
        """
        Generate a new solution by injecting a segment from the winner's routes.

        The loser adopts a contiguous segment of nodes from the winner's flat
        tour that it does not already service.  The resulting child is then
        repaired via greedy_insertion to maintain feasibility.

        Args:
            loser_routes: Losing solution's routes.
            winner_routes: Winning solution's routes.

        Returns:
            Child routing solution.
        """
        winner_flat = [n for r in winner_routes for n in r]
        loser_visited = {n for r in loser_routes for n in r}

        if len(winner_flat) < 2:
            return self._perturb(loser_routes)

        seg_len = max(1, len(winner_flat) // 4)
        start_idx = self.random.randint(0, max(0, len(winner_flat) - seg_len))
        segment = winner_flat[start_idx : start_idx + seg_len]
        new_nodes = [n for n in segment if n not in loser_visited]

        child = copy.deepcopy(loser_routes)

        # First remove some worst nodes to make room
        n_remove = min(len(new_nodes), max(1, self.params.perturbation_strength))
        with contextlib.suppress(Exception):
            if self.params.profit_aware_operators:
                child, _ = worst_profit_removal(child, n_remove, self.dist_matrix, self.wastes, self.R, self.C)
            else:
                child, _ = worst_removal(child, n_remove, self.dist_matrix)

        if new_nodes:
            with contextlib.suppress(Exception):
                if self.params.profit_aware_operators:
                    child = greedy_profit_insertion(
                        child,
                        new_nodes,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        self.R,
                        self.C,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=self.params.vrpp,
                    )
                else:
                    child = greedy_insertion(
                        child,
                        new_nodes,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=self.params.vrpp,
                    )

        # Apply comprehensive local search (reusing instance)
        return self.ls.optimize(child)

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes."""
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
