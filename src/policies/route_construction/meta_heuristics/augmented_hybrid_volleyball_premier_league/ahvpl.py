"""
Augmented Hybrid Volleyball Premier League (AHVPL) Solver.

Integrates four metaheuristic components into a unified optimization engine:
  1. ACO  — Ant Colony Optimization for intelligent heuristic initialization
  2. VPL  — Volleyball Premier League for macro-level population orchestration
  3. HGS  — Hybrid Genetic Search for diversity management and crossover
  4. ALNS — Adaptive Large Neighborhood Search for deep local search

The HGS integration replaces the standard VPL deterministic learning phase
with genetic crossover operators and bi-criteria fitness evaluation (profit +
diversity), preventing premature convergence.

Reference:
    Hybrid Volleyball Algorithm Research (reports/).
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.crossover_recombination import ordered_crossover
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns import (
    ALNSSolver,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.solver import (
    KSparseACOSolver,
)
from logic.src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.params import (
    AHVPLParams,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.evolution import (
    evaluate,
    update_biased_fitness,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import (
    Individual,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit


class AHVPLSolver:
    """
    Augmented Hybrid Volleyball Premier League solver for VRP variants.

    Combines ACO-driven initialization, VPL population management, HGS
    diversity-driven crossover/mutation, and ALNS deep local search.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: AHVPLParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        self.dist_matrix = np.array(dist_matrix)
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # ACO: construction + pheromone guidance
        self.aco_solver = KSparseACOSolver(dist_matrix, wastes, capacity, R, C, params.aco_params, mandatory_nodes)
        self.pheromone = self.aco_solver.pheromone
        self.constructor = self.aco_solver.constructor

        # ALNS: deep local search (coaching)
        self.alns_solver = ALNSSolver(dist_matrix, wastes, capacity, R, C, params.alns_params, mandatory_nodes)

        # HGS: LinearSplit for giant tour decoding
        self.split_manager = LinearSplit(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            params.hgs_params.max_vehicles,
            mandatory_nodes,
            vrpp=params.vrpp,
        )

    def _initialize_population(self) -> List[Individual]:
        population = []
        for _ in range(self.params.n_teams):
            ind = self._construct_individual()
            if ind:
                population.append(ind)
        return population

    def _construct_individual(self) -> Optional[Individual]:
        routes = self.constructor.construct()
        if not routes:
            return None

        ind = Individual(self._routes_to_giant_tour(routes), expand_pool=self.params.vrpp)
        ind.routes = [r[:] for r in routes]

        # Calculate precise cost/rev directly from ACO routes
        rev = sum(self.wastes.get(n, 0) for r in routes for n in r) * self.R
        cost = 0.0
        for r in routes:
            if not r:
                continue
            d = self.dist_matrix[0][r[0]]
            for i in range(len(r) - 1):
                d += self.dist_matrix[r[i]][r[i + 1]]
            d += self.dist_matrix[r[-1]][0]
            cost += d * self.C

        ind.cost = cost
        ind.revenue = rev
        ind.profit_score = rev - cost

        return ind

    def solve(self) -> Tuple[List[List[int]], float, float]:  # noqa: C901
        """
        Run the Augmented HVPL algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        start_time = time.process_time()

        # ── Phase 1: ACO-Driven Heuristic Initialization ──
        population = self._initialize_population()

        if not population:
            return [], 0.0, 0.0

        best_ind = max(population, key=lambda x: x.profit_score)
        best_routes = [r[:] for r in best_ind.routes]
        best_profit = best_ind.profit_score
        best_cost = best_ind.cost

        # ── Phase 2+3: VPL + HGS + ALNS Main Loop ──
        last_improvement_it = 0
        for _iteration in range(self.params.hgs_params.n_iterations_no_improvement):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # 1. Bi-criteria fitness for parent selection
            update_biased_fitness(
                population,
                self.params.hgs_params.nb_elite,
                self.params.hgs_params.nb_granular,
            )

            # 2. HGS Crossover — generate children (cheap)
            n_crossovers = max(1, int(len(population) * self.params.hgs_params.crossover_rate))
            n_children = 0
            for _ in range(n_crossovers):
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break
                p1, p2 = self._select_parents(population)
                child = self._active_crossover(p1, p2)

                # 2.5 Mutation: SWAP on giant tour
                if self.random.random() < self.params.hgs_params.mutation_rate:
                    self._mutate(child)

                evaluate(child, self.split_manager)
                population.append(child)
                n_children += 1

            # 3. Population-wide ALNS Coaching
            population.sort(key=lambda x: (x.profit_score, tuple(tuple(r) for r in x.routes)), reverse=True)
            for i, ind in enumerate(population):
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break

                if i < self.params.hgs_params.nb_elite:
                    # ULTRA REFINEMENT: pushes routing efficiency to the max
                    iters = self.params.elite_alns_iterations
                elif not ind.is_coached:
                    # New children
                    iters = self.params.not_coached_alns_iterations
                else:
                    # Base coaching
                    iters = self.params.alns_params.max_iterations

                population[i] = self._alns_coaching(ind, iterations=iters)

            # 4. Survivor Selection — trim back to n_teams.
            update_biased_fitness(
                population,
                self.params.n_teams,
                self.params.hgs_params.nb_granular,
            )
            population.sort(key=lambda x: x.fitness)
            population = population[: self.params.n_teams]

            # 5. Substitution Phase: Replace weakest teams with fresh ACO solutions
            # Increased rotation to 50% to maximize search breadth
            population.sort(key=lambda x: x.profit_score, reverse=True)
            n_sub = int(self.params.n_teams * self.params.sub_rate)
            for i in range(self.params.n_teams - n_sub, self.params.n_teams):
                new_ind = self._construct_individual()
                if new_ind:
                    population[i] = new_ind

            # 6. Update global best
            iter_best = max(population, key=lambda x: x.profit_score)
            if iter_best.profit_score > best_profit:
                best_routes = [r[:] for r in iter_best.routes]
                best_profit = iter_best.profit_score
                best_cost = iter_best.cost
                last_improvement_it = 0
            else:
                last_improvement_it += 1

            # 7. Pheromone Update — ACO global guidance based on best cost
            self._update_pheromones(best_routes, best_profit, best_cost)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iter_best.profit_score,
                population_size=len(population),
                n_children=n_children,
            )

        return best_routes, best_profit, best_cost

    # ── HGS Operators ────────────────────────────────────────────────

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Binary tournament selection."""

        def tournament() -> Individual:
            i1, i2 = self.random.sample(population, 2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()

    def _active_crossover(self, p1: Individual, p2: Individual) -> Individual:
        """
        Ordered Crossover on active (visited) nodes only.

        Uses the union of both parents' active node sets so both
        temporary individuals are permutations of the same set
        (required by OX). Each parent's ordering is preserved for its
        own active nodes; the other parent's exclusive nodes are appended.
        """
        # Identify active nodes from each parent's routes
        p1_active = set(n for r in p1.routes for n in r)
        p2_active = set(n for r in p2.routes for n in r)

        # Both parents must have active nodes for crossover
        if not p1_active or not p2_active:
            return Individual(p1.giant_tour[:])

        # Build each temp individual as a permutation of the union.
        # Keep original ordering for own active nodes, append the rest.
        p1_ordered = [n for n in p1.giant_tour if n in p1_active]
        p1_extra = [n for n in p2.giant_tour if n in (p2_active - p1_active)]
        p1_ordered.extend(p1_extra)

        p2_ordered = [n for n in p2.giant_tour if n in p2_active]
        p2_extra = [n for n in p1.giant_tour if n in (p1_active - p2_active)]
        p2_ordered.extend(p2_extra)

        temp_p1 = Individual(p1_ordered, expand_pool=p1.expand_pool)
        temp_p2 = Individual(p2_ordered, expand_pool=p2.expand_pool)
        child = ordered_crossover(temp_p1, temp_p2, rng=self.random)

        # Re-append globally unvisited nodes for genetic consistency
        visited = set(child.giant_tour)
        missing = [n for n in self.nodes if n not in visited]
        child.giant_tour.extend(missing)

        return child

    def _mutate(self, ind: Individual) -> None:
        """Apply SWAP mutation to giant tour."""
        size = len(ind.giant_tour)
        if size < 2:
            return
        idx1, idx2 = self.random.sample(range(size), 2)
        ind.giant_tour[idx1], ind.giant_tour[idx2] = ind.giant_tour[idx2], ind.giant_tour[idx1]
        ind.is_coached = False

    def _alns_coaching(self, ind: Individual, iterations: int = 100) -> Individual:
        """
        Apply ALNS deep local search to an individual's routes.
        """
        # Temporarily override iterations for this coaching session
        old_iters = self.alns_solver.params.max_iterations
        self.alns_solver.params.max_iterations = iterations

        improved_routes, improved_profit, improved_cost = self.alns_solver.solve(initial_solution=ind.routes)

        self.alns_solver.params.max_iterations = old_iters

        if improved_profit > ind.profit_score and improved_routes:
            ind.routes = [r[:] for r in improved_routes]
            ind.profit_score = improved_profit
            ind.cost = improved_cost

            # Update giant tour but DO NOT re-evaluate!
            # Re-evaluating with LinearSplit after concatenating routes
            # can sub-optimally destroy ALNS profit gains.
            ind.giant_tour = self._routes_to_giant_tour(improved_routes)

            # Preserve unvisited nodes for genetic consistency
            visited = set(ind.giant_tour)
            missing = [n for n in self.nodes if n not in visited]
            ind.giant_tour.extend(missing)

        # Mark as coached to skip in future iterations unless modified
        ind.is_coached = True
        return ind

    # ── ACO Pheromone Management ─────────────────────────────────────

    def _update_pheromones(self, routes: List[List[int]], profit: float, cost: float) -> None:
        if not routes or profit <= 0:
            return

        self.pheromone.evaporate_all(self.params.aco_params.rho)

        # Higher profit → stronger reinforcement
        delta = self.params.aco_params.elitist_weight * profit / (cost + 1e-6)
        for route in routes:
            if not route:
                continue
            self.pheromone.deposit_edge(0, route[0], delta)
            for k in range(len(route) - 1):
                self.pheromone.deposit_edge(route[k], route[k + 1], delta)
            self.pheromone.deposit_edge(route[-1], 0, delta)

    # ── Utility ──────────────────────────────────────────────────────

    @staticmethod
    def _routes_to_giant_tour(routes: List[List[int]]) -> List[int]:
        """Flatten routes into a single giant tour (node sequence)."""
        gt: List[int] = []
        for route in routes:
            gt.extend(route)
        return gt
