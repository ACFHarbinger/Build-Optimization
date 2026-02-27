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

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..adaptive_large_neighborhood_search.alns import ALNSSolver
from ..ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
from ..hybrid_genetic_search.evolution import (
    evaluate,
    ordered_crossover,
    update_biased_fitness,
)
from ..hybrid_genetic_search.individual import Individual
from ..hybrid_genetic_search.split import LinearSplit
from .params import AHVPLParams


class AHVPLSolver(PolicyVizMixin):
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
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Augmented HVPL algorithm.

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        start_time = time.time()

        # ── Phase 1: ACO-Driven Heuristic Initialization ──
        population = self._initialize_population()

        if not population:
            return [], 0.0, 0.0

        best_ind = max(population, key=lambda x: x.profit_score)
        best_routes = [r[:] for r in best_ind.routes]
        best_profit = best_ind.profit_score
        best_cost = best_ind.cost

        # ── Phase 2+3: VPL + HGS + ALNS Main Loop ──
        for _iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            # 1. Bi-criteria fitness for parent selection
            update_biased_fitness(population, self.params.hgs_params.elite_size)

            # 2. HGS Crossover — generate children (cheap)
            n_crossovers = max(1, int(len(population) * self.params.hgs_params.crossover_rate))
            n_children = 0
            for _ in range(n_crossovers):
                if time.time() - start_time > self.params.time_limit:
                    break
                p1, p2 = self._select_parents(population)
                child = self._active_crossover(p1, p2)
                evaluate(child, self.split_manager)
                population.append(child)
                n_children += 1

            # 3. Population-wide ALNS Coaching (on full pop including children)
            for i, ind in enumerate(population):
                if time.time() - start_time > self.params.time_limit:
                    break
                if ind.routes:
                    population[i] = self._alns_coaching(ind)

            # 4. Survivor Selection — trim by biased fitness
            update_biased_fitness(population, self.params.hgs_params.elite_size)
            population.sort(key=lambda x: x.fitness)
            population = population[: self.params.n_teams]

            # 5. Update global best
            iter_best = max(population, key=lambda x: x.profit_score)
            if iter_best.profit_score > best_profit:
                best_routes = [r[:] for r in iter_best.routes]
                best_profit = iter_best.profit_score
                best_cost = iter_best.cost

            # 6. Pheromone Update — ACO global guidance
            self._update_pheromones(best_routes, best_cost)

            self._viz_record(
                iteration=_iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iter_best.profit_score,
                population_size=len(population),
                n_children=n_children,
            )

            # 7. VPL Substitution — replace worst teams with fresh ACO solutions
            n_sub = max(1, int(self.params.n_teams * self.params.sub_rate))
            for i in range(len(population) - n_sub, len(population)):
                new_ind = self._construct_individual()
                if new_ind is not None:
                    population[i] = new_ind

        return best_routes, best_profit, best_cost

    # ── Initialization ──────────────────────────────────────────────

    def _initialize_population(self) -> List[Individual]:
        """Generate initial population using ACO constructor."""
        population: List[Individual] = []
        for _ in range(self.params.n_teams):
            ind = self._construct_individual()
            if ind is not None:
                population.append(ind)
        return population

    def _construct_individual(self) -> Optional[Individual]:
        """Build one Individual from an ACO-constructed solution."""
        routes = self.constructor.construct()
        if not routes:
            return None

        giant_tour = self._routes_to_giant_tour(routes)
        if not giant_tour:
            return None

        # Ensure all nodes are present for genetic consistency
        visited = set(giant_tour)
        missing = [n for n in self.nodes if n not in visited]
        giant_tour.extend(missing)

        ind = Individual(giant_tour)
        evaluate(ind, self.split_manager)
        return ind

    # ── HGS Operators ────────────────────────────────────────────────

    def _select_parents(self, population: List[Individual]) -> Tuple[Individual, Individual]:
        """Binary tournament selection."""

        def tournament() -> Individual:
            i1, i2 = random.sample(population, 2)
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

        temp_p1 = Individual(p1_ordered)
        temp_p2 = Individual(p2_ordered)
        child = ordered_crossover(temp_p1, temp_p2)

        # Re-append globally unvisited nodes for genetic consistency
        visited = set(child.giant_tour)
        missing = [n for n in self.nodes if n not in visited]
        child.giant_tour.extend(missing)

        return child

    def _alns_coaching(self, ind: Individual) -> Individual:
        """
        Apply ALNS deep local search to an individual's routes.

        Runs the ALNS solver starting from the individual's decoded routes,
        then reconstructs the giant tour and re-evaluates via LinearSplit
        for consistent fitness metrics.
        """
        improved_routes, improved_profit, improved_cost = self.alns_solver.solve(initial_solution=ind.routes)

        if improved_profit > ind.profit_score and improved_routes:
            ind.giant_tour = self._routes_to_giant_tour(improved_routes)

            # Preserve unvisited nodes for genetic consistency
            visited = set(ind.giant_tour)
            missing = [n for n in self.nodes if n not in visited]
            ind.giant_tour.extend(missing)

            # Re-evaluate through LinearSplit for consistent metrics
            evaluate(ind, self.split_manager)

        return ind

    # ── ACO Pheromone Management ─────────────────────────────────────

    def _update_pheromones(self, routes: List[List[int]], cost: float) -> None:
        """ACS-style global pheromone update on best solution's edges."""
        if not routes or cost <= 0:
            return

        self.pheromone.evaporate_all(self.params.aco_params.rho)

        delta = self.params.aco_params.elitist_weight / cost
        for route in routes:
            if not route:
                continue
            self.pheromone.update_edge(0, route[0], delta, evaporate=False)
            for k in range(len(route) - 1):
                self.pheromone.update_edge(route[k], route[k + 1], delta, evaporate=False)
            self.pheromone.update_edge(route[-1], 0, delta, evaporate=False)

    # ── Utility ──────────────────────────────────────────────────────

    @staticmethod
    def _routes_to_giant_tour(routes: List[List[int]]) -> List[int]:
        """Flatten routes into a single giant tour (node sequence)."""
        gt: List[int] = []
        for route in routes:
            gt.extend(route)
        return gt
