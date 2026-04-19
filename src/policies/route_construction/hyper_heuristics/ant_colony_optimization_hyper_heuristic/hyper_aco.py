"""
Hyper-Heuristic Ant Colony Optimization (Hyper-ACO).

This module implements a Hyper-Heuristic ACO where ants construct
sequences of local search operators rather than node sequences.
The pheromone matrix encodes transition probabilities between operators.

Key Features:
- Operator graph: Nodes represent local search operators (2-opt, swap, etc.)
- Pheromone trail: tau[i][j] = favorability of applying operator j after operator i
- Heuristic information: Dynamic based on operator success rate
- Solution evaluation: Apply operator sequence to a base solution

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_hyper_heuristic import HyperHeuristicACO
    >>> solver = HyperHeuristicACO(dist_matrix, wastes, capacity, R, C, params)
    >>> best_solution = solver.solve(initial_solution)
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .hyper_operators import (
    HYPER_OPERATORS,
    HyperOperatorContext,
)
from .params import HyperACOParams


class HyperHeuristicACO:
    """
    Hyper-Heuristic ACO solver.

    Ants construct sequences of local search operators.
    The best sequences are reinforced via pheromone updates.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Optional[HyperACOParams] = None,
        initial_solution: Optional[List[List[int]]] = None,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize HyperHeuristicACO.

        Args:
            dist_matrix: Distance matrix between nodes.
            wastes: Dictionary mapping node indices to wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Hyper-heuristics parameters object.
            initial_solution: Optional starting routes.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params or HyperACOParams()
        self.initial_solution = initial_solution or []
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(self.params.seed) if self.params.seed is not None else random.Random()
        self.np_rng = (
            np.random.default_rng(self.params.seed) if self.params.seed is not None else np.random.default_rng()
        )

        self.operator_names = list(HYPER_OPERATORS.keys())
        self.n_operators = len(self.operator_names)
        self.op_to_idx = {name: i for i, name in enumerate(self.operator_names)}

        # Task 3: Dynamic journey length matches number of heuristics
        self.sequence_length = self.n_operators

        # Pheromone matrix: n_operators x n_operators
        # tau[i][j] where i is the previous operator and j is the next.
        # We add one extra row for the start state (0).
        self.tau = np.full((self.n_operators + 1, self.n_operators), self.params.tau_0)

        # Heuristic information: Time-weighted visibility (Task 1)
        # eta[i][j] represents visibility of transitioning from operator i to j
        self.eta = np.ones((self.n_operators + 1, self.n_operators))

        # Edge traversal counter: num(i,j) tracks how many times edge (i,j) was traversed
        self.edge_count = np.zeros((self.n_operators + 1, self.n_operators))

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Solve VRP using Hyper-ACO with proper ant state synchronization.

        Task 1: Accumulate eta updates from all ants and apply evaporation once per iteration.
        Task 2: Maintain individual ant solutions, only synchronize to global best when improved.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
        """
        best_routes = copy.deepcopy(self.initial_solution)
        best_cost = self._calculate_cost(best_routes)
        best_profit = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r) - best_cost

        # Task 2: Initialize individual ant solutions (swarm memory)
        ant_solutions = [copy.deepcopy(self.initial_solution) for _ in range(self.params.n_ants)]

        start_time = time.process_time()
        for _it in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.process_time() - start_time > self.params.time_limit:
                break

            # Task 1: Initialize eta accumulator for this iteration
            total_eta_updates = np.zeros_like(self.eta)

            # 1. Ant solutions construction - each ant works from its own state
            ant_results: List[Tuple[List[List[int]], float, float, List[str], int]] = []

            for ant_idx in range(self.params.n_ants):
                start_cost = self._calculate_cost(ant_solutions[ant_idx])
                routes, sequence, eta_updates = self.build_solution(ant_solutions[ant_idx])
                final_cost = self._calculate_cost(routes)

                # Task 1: Accumulate eta updates from this ant
                total_eta_updates += eta_updates

                # Update this ant's individual solution
                ant_solutions[ant_idx] = routes

                ant_results.append((routes, start_cost, final_cost, sequence, ant_idx))

            # 2. Identify iteration best by final cost
            iter_best_routes = min(ant_results, key=lambda x: x[2])[0]
            iter_best_cost = self._calculate_cost(iter_best_routes)
            iter_best_profit = sum(self.wastes.get(n, 0) * self.R for r in iter_best_routes for n in r) - iter_best_cost

            # Task 2: Synchronization - only if new global best found
            global_best_improved = False
            if iter_best_profit > best_profit:
                best_profit = iter_best_profit
                best_cost = iter_best_cost
                best_routes = copy.deepcopy(iter_best_routes)
                global_best_improved = True

                # Synchronize all ants to the new global best
                ant_solutions = [copy.deepcopy(best_routes) for _ in range(self.params.n_ants)]

            # 3. Pheromone evaporation
            self._evaporate_pheromones()

            # 4. Multi-ant pheromone deposit (AS-style, not ACS elitist)
            # Only ants with positive journey improvement deposit pheromone
            for _routes, start_cost_ant, final_cost_ant, sequence, _ant_idx in ant_results:
                journey_improvement = start_cost_ant - final_cost_ant  # I_k
                if journey_improvement > 0:
                    # Delta_tau = I_k / L_k
                    delta_tau = journey_improvement / len(sequence)
                    prev_op_idx = self.n_operators  # Start state
                    for op_name in sequence:
                        next_op_idx = self.op_to_idx[op_name]
                        self.tau[prev_op_idx][next_op_idx] += delta_tau
                        prev_op_idx = next_op_idx

            # 5. Clip pheromones after deposit
            np.clip(self.tau, self.params.tau_min, self.params.tau_max, out=self.tau)

            # Task 1: Apply visibility evaporation and updates ONCE per iteration
            self.eta = (self.params.rho * self.eta) + total_eta_updates
            self.eta = np.clip(self.eta, 0.01, 100.0)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_it,
                best_profit=best_profit,
                best_cost=best_cost,
                iter_best_cost=iter_best_cost,
                tau_mean=float(self.tau.mean()),
                eta_mean=float(self.eta.mean()),
                global_best_improved=global_best_improved,
            )

        collected_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        final_cost = best_cost / self.C if self.C > 0 else best_cost
        return best_routes, collected_rev - best_cost, final_cost

    def build_solution(self, base_solution: List[List[int]]) -> Tuple[List[List[int]], List[str], np.ndarray]:
        """
        Build a solution by applying a sequence of operators.

        Task 1: Track execution time, cost improvement, and return eta updates.
        The visibility matrix evaporation is applied once per iteration, not per ant.

        Args:
            base_solution: The starting solution for this ant.

        Returns:
            Tuple[List[List[int]], List[str], np.ndarray]: Modified routes, operator sequence, and eta updates.
        """
        sequence = self._select_sequence()
        ctx = HyperOperatorContext(
            routes=copy.deepcopy(base_solution),
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
            profit_aware_operators=self.params.profit_aware_operators,
            vrpp=self.params.vrpp,
        )

        # Task 1: Time-weighted visibility update
        # Accumulate visibility updates for this ant's journey
        eta_updates = np.zeros_like(self.eta)
        prev_op_idx = self.n_operators  # Start state

        for op_name in sequence:
            next_op_idx = self.op_to_idx[op_name]
            op_func = HYPER_OPERATORS.get(op_name)

            if op_func:
                # Measure execution time T_kj(t)
                cost_before = self._calculate_cost(ctx.routes)
                start_time = time.process_time()
                op_func(ctx)
                execution_time = time.process_time() - start_time
                cost_after = self._calculate_cost(ctx.routes)

                # Calculate cost improvement I_kj
                cost_improvement = cost_before - cost_after

                # Apply monotonic conversion
                lambda_power = self.params.lambda_factor**cost_improvement

                # Increment edge count
                self.edge_count[prev_op_idx][next_op_idx] += 1

                # Compute visibility contribution: lambda^I_kj / (T_kj * num(i,j))
                # Avoid division by zero
                safe_time = max(execution_time, 1e-9)
                safe_count = max(self.edge_count[prev_op_idx][next_op_idx], 1)

                eta_updates[prev_op_idx][next_op_idx] += lambda_power / (safe_time * safe_count)

            prev_op_idx = next_op_idx

        # Return eta_updates without applying them - solve() will handle the evaporation
        return ctx.routes, sequence, eta_updates

    def _select_sequence(self) -> List[str]:
        """
        Construct an operator sequence using ACO rules with ACS q0 exploitation.

        Task 3: Use self.sequence_length (dynamically set to n_operators).
        """
        sequence = []
        current_op_idx = self.n_operators  # Start state (index n_operators in tau)

        for _ in range(self.sequence_length):
            # ACS pseudo-random proportional rule with 2D eta
            numerator = (self.tau[current_op_idx] ** self.params.alpha) * (self.eta[current_op_idx] ** self.params.beta)

            # ACS q0 rule: exploit vs. explore
            if self.random.random() < self.params.q0:
                # Exploit: choose best known operator
                next_op_idx = int(np.argmax(numerator))
            else:
                # Explore: roulette wheel selection
                denom = np.sum(numerator)
                probs = numerator / denom if denom > 1e-12 else np.ones(self.n_operators) / self.n_operators
                next_op_idx = self.np_rng.choice(self.n_operators, p=probs)

            sequence.append(self.operator_names[next_op_idx])
            current_op_idx = next_op_idx

        return sequence

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0, route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i], route[i + 1]]
            total += self.dist_matrix[route[-1], 0]
        return total * self.C

    def _evaporate_pheromones(self):
        """Apply pheromone evaporation (no clipping here per Phase 4)."""
        self.tau *= 1 - self.params.rho
