"""
Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL) Solver.

Combines:
    1. Enhanced ACO with Q-Learning for intelligent solution construction
    2. Enhanced ALNS with SARSA for adaptive destroy/repair operator selection
    3. Population-based VPL framework for global search and diversity

This is a middle-ground between:
    - HVPL: Basic ACO + ALNS in population framework
    - RL-AHVPL: Full HGS with genetic operators, CMAB, GLS, reactive tabu

RL-HVPL provides RL-enhanced operators without genetic complexity.

Reference:
    Volleyball Premier League Algorithm (VPL)
    ACO with Q-Learning for dynamic operator selection
    ALNS with SARSA for adaptive search
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.reinforcement_learning.alns_sarsa import (
    ALNSSARSASolver,
)
from logic.src.policies.helpers.reinforcement_learning.ks_aco_qlearning import (
    KSparseACOQLSolver,
)

from .params import RLHVPLParams


class RLHVPLSolver:
    """
    Reinforcement Learning Hybrid Volleyball Premier League solver for VRP variants.

    Architecture:
        - Population (Teams): Multiple candidate solutions competing
        - Construction (ACO-Q): Q-Learning guides ACO's local search operator selection
        - Coaching (ALNS-SARSA): SARSA guides ALNS's destroy/repair operator selection
        - Competition: Best teams survive, weakest are replaced
        - Global Guidance: Pheromone updates based on best solutions
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RLHVPLParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize RL-HVPL solver.

        Args:
            dist_matrix: Distance matrix between nodes (including depot at index 0).
            wastes: Dictionary mapping node IDs to waste amounts.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: RLHVPLParams configuration.
            mandatory_nodes: Nodes that must be visited (if any).
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []

        # Initialize enhanced ACO with Q-Learning
        self.aco_solver = KSparseACOQLSolver(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            params.aco_params,
            rl_params=params,
            mandatory_nodes=mandatory_nodes,
        )
        self.pheromone = self.aco_solver.pheromone
        self.constructor = self.aco_solver.constructor

        # Initialize enhanced ALNS with SARSA
        self.alns_solver = ALNSSARSASolver(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            params.alns_params,
            params,
            mandatory_nodes,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the RL-HVPL algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: (best_routes, best_profit, best_cost)
        """
        start_time = time.process_time()

        # ===== Phase 1: Initialization - Create Initial Population =====
        population: List[Tuple[List[List[int]], float, float]] = []
        for _ in range(self.params.n_teams):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Use enhanced ACO with Q-Learning to construct solution
            routes, profit, cost = self.aco_solver.solve()
            routes = self._canonicalize_routes(routes)
            population.append((routes, profit, cost))

        # Handle empty population edge case
        if not population:
            return [], 0.0, 0.0

        # Initialize best solution
        best_routes, best_profit, best_cost = self._get_best(population)

        # ===== Phase 2: Main Evolutionary Loop =====
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # ===== 3. Coaching Phase: Apply ALNS-SARSA to each team =====
            new_population = []
            for team_rank, (routes, _profit, _cost) in enumerate(population):
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break

                # Determine coaching intensity based on team rank
                if team_rank < self.params.elite_size:
                    # Elite teams get intensive coaching
                    coaching_iters = self.params.elite_coaching_iterations
                else:
                    # Regular teams get lighter coaching
                    coaching_iters = self.params.regular_coaching_iterations

                # Apply ALNS-SARSA coaching
                coached_routes, coached_profit, coached_cost = self._apply_coaching(routes, coaching_iters)
                coached_routes = self._canonicalize_routes(coached_routes)
                new_population.append((coached_routes, coached_profit, coached_cost))

            population = new_population

            # ===== 4. Competition: Update Global Best =====
            iter_best_routes, iter_best_profit, iter_best_cost = self._get_best(population)
            if iter_best_profit > best_profit:
                best_routes = copy.deepcopy(iter_best_routes)
                best_profit = iter_best_profit
                best_cost = iter_best_cost

            # ===== 5. Pheromone Update: Global Guidance =====
            # Use profit-based pheromone updates to bias toward high-value solutions
            self._update_pheromones(best_routes, best_profit, best_cost)

            # ===== 6. Substitution: Replace Weakest Teams =====
            # Sort by profit (descending), then cost (ascending), then deterministic hash
            population.sort(key=lambda x: (x[1], -x[2], self._hash_routes(x[0])), reverse=True)
            n_sub = int(self.params.n_teams * self.params.sub_rate)

            for i in range(self.params.n_teams - n_sub, self.params.n_teams):
                if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                    break

                # Replace with new solution generated using updated pheromones
                new_routes, new_profit, new_cost = self.aco_solver.solve()
                new_routes = self._canonicalize_routes(new_routes)
                population[i] = (new_routes, new_profit, new_cost)

            # ===== Visualization & Logging =====
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_profit=iter_best_profit,
                population_size=len(population),
                aco_epsilon=self.aco_solver.agent.epsilon,
                alns_epsilon=self.alns_solver.agent.epsilon,
            )

        return best_routes, best_profit, best_cost

    # ===== Helper Methods =====

    def _apply_coaching(self, routes: List[List[int]], iterations: int) -> Tuple[List[List[int]], float, float]:
        """
        Apply ALNS-SARSA coaching to improve a solution.

        Args:
            routes: Current solution routes.
            iterations: Number of ALNS iterations to perform.

        Returns:
            Tuple[List[List[int]], float, float]: (improved_routes, profit, cost)
        """
        # Temporarily override ALNS max_iterations
        old_iters = self.alns_solver.params.max_iterations
        self.alns_solver.params.max_iterations = iterations

        # Run ALNS-SARSA
        improved_routes, improved_profit, improved_cost = self.alns_solver.solve(initial_solution=routes)

        # Restore original parameter
        self.alns_solver.params.max_iterations = old_iters

        return improved_routes, improved_profit, improved_cost

    def _canonicalize_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Canonicalize routes by sorting them by first node for consistent ordering.

        Args:
            routes: List of routes to canonicalize.

        Returns:
            List[List[int]]: Sorted routes.
        """
        return sorted([r for r in routes if r], key=lambda x: x[0] if x else 0)

    def _hash_routes(self, routes: List[List[int]]) -> str:
        """
        Create a deterministic hash of routes for tie-breaking.

        Args:
            routes: Routes to hash.

        Returns:
            str: Hash string.
        """
        sorted_routes = self._canonicalize_routes(routes)
        return "|".join(",".join(map(str, r)) for r in sorted_routes)

    def _get_best(self, population: List[Tuple[List[List[int]], float, float]]) -> Tuple[List[List[int]], float, float]:
        """
        Get the best solution from the population.

        Uses deterministic tie-breaking: profit (desc), cost (asc), hash (asc).

        Args:
            population: List of (routes, profit, cost) tuples.

        Returns:
            Tuple[List[List[int]], float, float]: Best (routes, profit, cost).
        """
        return max(
            population,
            key=lambda x: (x[1], -x[2], self._hash_routes(x[0])),
        )

    def _update_pheromones(self, routes: List[List[int]], profit: float, cost: float) -> None:
        """
        Update pheromones based on the best solution.

        Strategy:
            - Profit-based: Deposit pheromone proportional to profit/cost ratio
            - Cost-based: Deposit pheromone inversely proportional to cost

        Args:
            routes: Best routes to reinforce.
            profit: Profit of the best solution.
            cost: Cost of the best solution.
        """
        if not routes:
            return

        # Evaporate existing pheromones
        self.pheromone.evaporate_all(self.params.aco_params.rho)

        # Calculate pheromone deposit amount
        if self.params.pheromone_update_strategy == "profit":
            # Profit-based: reward high profit-to-cost ratio
            if cost > 0:
                delta = self.params.aco_params.elitist_weight * self.params.profit_weight * profit / cost
            else:
                delta = self.params.aco_params.elitist_weight * profit
        else:
            # Cost-based (classic ACS style)
            delta = self.params.aco_params.elitist_weight / cost if cost > 0 else self.params.aco_params.elitist_weight

        # Deposit pheromones on edges in the best routes
        for route in routes:
            if not route:
                continue

            # Depot to first node
            self.pheromone.deposit_edge(0, route[0], delta)

            # Between consecutive nodes
            for k in range(len(route) - 1):
                self.pheromone.deposit_edge(route[k], route[k + 1], delta)

            # Last node back to depot
            self.pheromone.deposit_edge(route[-1], 0, delta)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance cost.

        Args:
            routes: Routes to evaluate.

        Returns:
            float: Total distance traveled.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            # Depot to first node
            total += self.dist_matrix[0][route[0]]
            # Between nodes
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            # Last node to depot
            total += self.dist_matrix[route[-1]][0]
        return total
