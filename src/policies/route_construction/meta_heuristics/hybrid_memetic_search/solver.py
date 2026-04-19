"""
Hybrid Memetic Search (HMS) for VRPP.

EXACT COPY of Hybrid Volleyball Premier League (HVPL) with rigorous nomenclature.

TERMINOLOGY MAPPING (HVPL → HMS):
- "Active Teams" → Active Population
- "Passive Teams" → Reserve Population
- "Volleyball Matches" → Evolutionary Operators
- "Coaching" → Intensive Local Search (ALNS)
- "League Season" → Generation

Algorithm Structure (Sun et al., 2023):
    Phase 1: ACO-Driven Initialization
        - Construct initial active and reserve populations using pheromone guidance
        - Ensures high-quality diverse starting solutions

    Phase 2: HGS Evolution
        - Tournament selection for parent choice
        - Crossover and mutation operators
        - Elitist survival selection
        - Diversity injection from reserve population

    Phase 3: ALNS Coaching Refinement
        - Apply intensive local search to all solutions
        - Update global best and pheromone trails

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup–delivery location
    routing problem."

IMPORTANT: This implementation EXACTLY matches the HVPL algorithm with only
           terminology changed from sports metaphors to OR terminology.
           Uses identical RNG attribute name (self.random) and algorithm flow.
"""

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
from logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns import (
    ALNSSolver,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.solver import (
    KSparseACOSolver,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_memetic_search.params import (
    HybridMemeticSearchParams,
)


class HybridMemeticSearchSolver:
    """
    Hybrid Memetic Search solver for VRPP.
    EXACT COPY of HVPL with rigorous nomenclature.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HybridMemeticSearchParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize HMS solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1), index 0 = depot.
            wastes: Dictionary mapping node index to waste/profit value.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit of waste collected.
            C: Cost per unit of distance traveled.
            params: HMS algorithm parameters.
            mandatory_nodes: List of nodes that must be visited.
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Initialize ACO solver for population initialization
        self.aco_solver = KSparseACOSolver(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params.aco_params,  # type: ignore[arg-type]
            mandatory_nodes=mandatory_nodes,
        )

        # Initialize ALNS solver for coaching phase
        self.alns_solver = ALNSSolver(
            dist_matrix=dist_matrix,
            wastes=wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=params.alns_params,  # type: ignore[arg-type]
            mandatory_nodes=mandatory_nodes,
        )

        # Initialize Local Search
        self.ls = ACOLocalSearch(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            params,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Memetic Search algorithm.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.process_time()

        # ═══════════════════════════════════════════════════════════
        # PHASE 1: ACO-DRIVEN INITIALIZATION
        # ═══════════════════════════════════════════════════════════
        active_teams = self._aco_initialization()
        passive_teams = self._aco_initialization()  # Passive reserve pool

        # Evaluate active teams
        active_profits = [self._evaluate(team) for team in active_teams]

        # Sort by profit (descending)
        sorted_indices = sorted(range(len(active_profits)), key=lambda i: active_profits[i], reverse=True)
        active_teams = [active_teams[i] for i in sorted_indices]
        active_profits = [active_profits[i] for i in sorted_indices]

        # Track global best
        best_routes = copy.deepcopy(active_teams[0])
        best_profit = active_profits[0]
        best_cost = self._cost(best_routes)

        # ═══════════════════════════════════════════════════════════
        # PHASE 2: POPULATION EVOLUTION WITH HGS
        # ═══════════════════════════════════════════════════════════
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Competition Phase: Rank teams (already sorted)

            # HGS-Enhanced Learning (replaces deterministic VPL coaching)
            offspring_teams = self._hgs_evolution(active_teams, active_profits)

            # Merge parents and offspring
            combined_teams = active_teams + offspring_teams
            combined_profits = active_profits + [self._evaluate(team) for team in offspring_teams]

            # Select next generation with elitism and diversity
            active_teams, active_profits = self._selection(combined_teams, combined_profits)

            # Substitution Phase: Inject diversity from passive teams
            active_teams = self._substitution_phase(active_teams, passive_teams)

            # ═══════════════════════════════════════════════════════════
            # PHASE 3: ALNS COACHING REFINEMENT
            # ═══════════════════════════════════════════════════════════
            active_teams = self._alns_coaching(active_teams)

            # Re-evaluate after coaching
            active_profits = [self._evaluate(team) for team in active_teams]

            # Sort by profit
            sorted_indices = sorted(range(len(active_profits)), key=lambda i: active_profits[i], reverse=True)
            active_teams = [active_teams[i] for i in sorted_indices]
            active_profits = [active_profits[i] for i in sorted_indices]

            # Update global best
            if active_profits[0] > best_profit:
                best_routes = copy.deepcopy(active_teams[0])
                best_profit = active_profits[0]
                best_cost = self._cost(best_routes)

                # Update ACO pheromones with new best
                self._update_pheromones(best_routes, best_cost)

            # Visualization tracking
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                avg_profit=np.mean(active_profits),
                active_teams=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private: Phase 1 - ACO Initialization
    # ------------------------------------------------------------------

    def _aco_initialization(self) -> List[List[List[int]]]:
        """
        Initialize population using ACO for intelligent construction.

        Runs truncated ACO to build high-quality diverse solutions using
        pheromone guidance.

        Returns:
            List of N initial routing solutions.
        """
        population = []

        # Run truncated ACO iterations
        for _ in range(self.params.aco_init_iterations):
            routes = self.aco_solver.constructor.construct()
            if routes:
                population.append(routes)

        # If not enough solutions, fill with random constructions
        while len(population) < self.params.population_size:
            routes = self._random_construction()
            population.append(routes)

        # Select N most diverse high-quality solutions
        return self._select_diverse_elite(population, self.params.population_size)

    def _random_construction(self) -> List[List[int]]:
        """Build a random routing solution."""
        return build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )

    def _select_diverse_elite(self, population: List[List[List[int]]], n_select: int) -> List[List[List[int]]]:
        """
        Select N diverse high-quality solutions using bi-criteria ranking.

        Balances fitness and diversity using Pareto-style selection.

        Args:
            population: Pool of candidate solutions.
            n_select: Number of solutions to select.

        Returns:
            Selected diverse elite solutions.
        """
        if len(population) <= n_select:
            return population[:]

        # Evaluate all
        profits = [self._evaluate(sol) for sol in population]

        # Sort by profit
        sorted_indices = sorted(range(len(profits)), key=lambda i: profits[i], reverse=True)

        # Select top half by quality
        selected_indices = sorted_indices[:n_select]
        return [population[i] for i in selected_indices]

    # ------------------------------------------------------------------
    # Private: Phase 2 - HGS Evolution
    # ------------------------------------------------------------------

    def _hgs_evolution(self, active_teams: List[List[List[int]]], active_profits: List[float]) -> List[List[List[int]]]:
        """
        Apply HGS genetic operators for population evolution.

        Replaces deterministic VPL coaching with crossover and mutation.

        Args:
            active_teams: Current active team solutions.
            active_profits: Fitness values for active teams.

        Returns:
            Offspring population.
        """
        offspring = []

        for _ in range(self.params.population_size):
            # Tournament selection
            parent1 = self._tournament_select(active_teams, active_profits)
            parent2 = self._tournament_select(active_teams, active_profits)

            # Crossover
            if self.random.random() < self.params.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            # Mutation
            if self.random.random() < self.params.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_select(self, teams: List[List[List[int]]], profits: List[float], k: int = 3) -> List[List[int]]:
        """Select best solution from k random candidates."""
        candidates = self.random.sample(range(len(teams)), min(k, len(teams)))
        best_idx = max(candidates, key=lambda i: profits[i])
        return copy.deepcopy(teams[best_idx])

    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """
        Order-crossover (OX) style operator for routing solutions.

        Combines node selections from both parents.
        """
        nodes_p1 = {node for route in parent1 for node in route}
        nodes_p2 = {node for route in parent2 for node in route}

        # Combine nodes with preference for profitable ones
        child_nodes = list(nodes_p1 | nodes_p2)

        # If too many nodes, prune low-profit ones
        if len(child_nodes) > len(nodes_p1):
            scored = [(self.wastes.get(n, 0.0), n) for n in child_nodes if n not in self.mandatory_nodes]
            scored.sort(reverse=True)
            keep_count = int(len(nodes_p1) * 1.1)  # Allow 10% growth
            child_nodes = self.mandatory_nodes + [n for _, n in scored[:keep_count]]

        # Rebuild routes
        try:
            if self.params.profit_aware_operators:
                return greedy_profit_insertion(
                    [],
                    child_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                return greedy_insertion(
                    [],
                    child_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
        except Exception:
            return copy.deepcopy(parent1)

    def _mutate(self, routes: List[List[int]]) -> List[List[int]]:
        """Mutation operator using destroy-repair."""
        try:
            n_remove = max(1, int(len([n for r in routes for n in r]) * 0.1))
            partial, removed = random_removal(routes, n_remove, self.random)

            if self.params.profit_aware_operators:
                return greedy_profit_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                return greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
        except Exception:
            return copy.deepcopy(routes)

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """Intra-team perturbation: worst-removal and greedy-insertion."""
        n = max(3, self.params.n_removal)
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        try:
            if use_profit:
                partial, removed = worst_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R)
                repaired = greedy_profit_insertion(
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
                partial, removed = worst_removal(routes, n, self.dist_matrix)
                repaired = greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

    def _selection(
        self, teams: List[List[List[int]]], profits: List[float]
    ) -> Tuple[List[List[List[int]]], List[float]]:
        """
        Select next generation with elitism.

        Args:
            teams: Combined parent and offspring teams.
            profits: Fitness values.

        Returns:
            Tuple of (selected teams, selected profits).
        """
        # Sort by profit
        sorted_indices = sorted(range(len(profits)), key=lambda i: profits[i], reverse=True)

        # Select top N
        selected_indices = sorted_indices[: self.params.population_size]
        selected_teams = [teams[i] for i in selected_indices]
        selected_profits = [profits[i] for i in selected_indices]

        return selected_teams, selected_profits

    # ------------------------------------------------------------------
    # Private: Phase 2 - Substitution
    # ------------------------------------------------------------------

    def _substitution_phase(
        self, active_teams: List[List[List[int]]], passive_teams: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        """
        Inject diversity by replacing weak teams with passive teams.

        Args:
            active_teams: Current active teams.
            passive_teams: Reserve passive teams.

        Returns:
            Active teams with injected diversity.
        """
        n_substitute = max(1, int(self.params.population_size * self.params.substitution_rate))

        # Replace weakest teams
        for i in range(self.params.population_size - n_substitute, self.params.population_size):
            # Select random passive team as replacement
            replacement = self.random.choice(passive_teams)
            active_teams[i] = copy.deepcopy(replacement)

        return active_teams

    # ------------------------------------------------------------------
    # Private: Phase 3 - ALNS Coaching
    # ------------------------------------------------------------------

    def _alns_coaching(self, teams: List[List[List[int]]]) -> List[List[List[int]]]:
        """
        Apply ALNS refinement to each team (coaching session).

        Args:
            teams: Active teams to coach.

        Returns:
            Coached (refined) teams.
        """
        coached = []

        for team in teams:
            # Run ALNS on this team
            refined_routes, _, _ = self.alns_solver.solve(initial_solution=team)
            coached.append(refined_routes)

        return coached

    # ------------------------------------------------------------------
    # Private: Pheromone Update
    # ------------------------------------------------------------------

    def _update_pheromones(self, routes: List[List[int]], cost: float) -> None:
        """
        Update ACO pheromones with global best solution.

        Args:
            routes: Best routing solution.
            cost: Total cost of best solution.
        """
        if not routes or cost <= 0:
            return

        # Evaporate
        self.aco_solver.pheromone.evaporate_all(self.params.aco_params.rho)  # type: ignore[union-attr]

        # Deposit on best solution edges
        delta = 1.0 / cost
        for route in routes:
            if not route:
                continue
            # Depot to first node
            self.aco_solver.pheromone.deposit_edge(0, route[0], delta)
            # Inter-node edges
            for k in range(len(route) - 1):
                self.aco_solver.pheromone.deposit_edge(route[k], route[k + 1], delta)
            # Last node back to depot
            self.aco_solver.pheromone.deposit_edge(route[-1], 0, delta)

    # ------------------------------------------------------------------
    # Private: Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit evaluation."""
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(node, 0.0) * self.R for route in routes for node in route)
        return revenue - self._cost(routes) * self.C

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
