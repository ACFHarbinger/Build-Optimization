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
    >>> from logic.src.policies.ant_colony_optimization.hyper_heuristic_aco import HyperHeuristicACO
    >>> solver = HyperHeuristicACO(dist_matrix, wastes, capacity, R, C, params)
    >>> best_solution = solver.solve(initial_solution)
"""

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from .hyper_operators import (
    HYPER_OPERATORS,
    HyperOperatorContext,
)
from .params import HyperACOParams


class HyperHeuristicACO(PolicyVizMixin):
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

        self.operator_names = list(HYPER_OPERATORS.keys())
        self.n_operators = len(self.operator_names)
        self.op_to_idx = {name: i for i, name in enumerate(self.operator_names)}

        # Pheromone matrix: n_operators x n_operators
        # tau[i][j] where i is the previous operator and j is the next.
        # We add one extra row for the start state (0).
        self.tau = np.full((self.n_operators + 1, self.n_operators), self.params.tau_0)

        # Heuristic information: derived from success rates
        self.eta = np.ones(self.n_operators)
        self.success_counts = np.zeros(self.n_operators)
        self.use_counts = np.ones(self.n_operators)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Solve VRP using Hyper-ACO.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
        """
        best_routes = copy.deepcopy(self.initial_solution)
        best_cost = self._calculate_cost(best_routes)

        start_time = time.time()
        for _it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            ant_solutions = []
            for _ant in range(self.params.n_ants):
                routes = self.build_solution()
                cost = self._calculate_cost(routes)
                ant_solutions.append((routes, cost))

            # Sort by cost and pick best
            ant_solutions.sort(key=lambda x: x[1])
            iter_best_routes, iter_best_cost = ant_solutions[0]

            if iter_best_cost < best_cost:
                best_cost = iter_best_cost
                best_routes = copy.deepcopy(iter_best_routes)

            # Update pheromones
            self._evaporate_pheromones()
            # In Hyper-ACO, we deposit based on sequence quality.
            # This is simplified here.

            # Update Success-based Heuristics
            self._update_heuristics()

            self._viz_record(
                iteration=_it,
                best_cost=best_cost,
                iter_best_cost=iter_best_cost,
                tau_mean=float(self.tau.mean()),
                eta_mean=float(self.eta.mean()),
            )

        collected_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        return best_routes, collected_rev - best_cost, best_cost / self.C if self.C > 0 else best_cost

    def build_solution(self) -> List[List[int]]:
        """
        Build a solution by applying a sequence of operators.
        """
        sequence = self._select_sequence()
        ctx = HyperOperatorContext(
            routes=copy.deepcopy(self.initial_solution),
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

        for op_name in sequence:
            op_func = HYPER_OPERATORS.get(op_name)
            if op_func:
                improved = op_func(ctx)
                op_idx = self.op_to_idx.get(op_name, 0)
                self.use_counts[op_idx] += 1
                if improved:
                    self.success_counts[op_idx] += 1

        return ctx.routes

    def _select_sequence(self) -> List[str]:
        """Construct an operator sequence using ACO rules."""
        sequence = []
        current_op_idx = self.n_operators  # Start state

        for _ in range(self.params.sequence_length):
            # Proportional selection logic
            probs = (self.tau[current_op_idx] ** self.params.alpha) * (self.eta**self.params.beta)
            probs /= np.sum(probs)

            next_op_idx = np.random.choice(self.n_operators, p=probs)
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
        """Apply pheromone evaporation."""
        self.tau *= 1 - self.params.rho
        np.clip(self.tau, self.params.tau_min, self.params.tau_max, out=self.tau)

    def _update_heuristics(self):
        """Update heuristic information based on operator success rates."""
        # eta = success rate with a floor
        self.eta = self.success_counts / self.use_counts
        self.eta = np.clip(self.eta, 0.01, 10.0)
