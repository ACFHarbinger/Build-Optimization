"""
Adaptive Large Neighborhood Search (ALNS) policy module.

This module provides the main entry points for the ALNS metaheuristic,
dispatching to specialized implementations based on configuration.

Reference:
    Pisinger, D., & Ropke, S. (2007). A general heuristic for vehicle routing problems.
    Computers & Operations Research, 34(8), 2403-2435.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators.destroy_operators import cluster_removal, random_removal, worst_removal
from ..operators.repair_operators import greedy_insertion, regret_2_insertion
from .alns_package import run_alns_package
from .ortools_wrapper import run_alns_ortools
from .params import ALNSParams


class ALNSSolver(PolicyVizMixin):
    """
    Custom implementation of Adaptive Large Neighborhood Search for CVRP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ALNSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the ALNS solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: Detailed ALNS parameters.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        # Operator registry
        self.destroy_ops = [
            lambda r, n: random_removal(r, n),
            lambda r, n: worst_removal(r, n, self.dist_matrix),
            lambda r, n: cluster_removal(r, n, self.dist_matrix, self.nodes),
        ]
        self.repair_ops = [
            lambda r, n: greedy_insertion(
                r, n, self.dist_matrix, self.wastes, self.capacity, R=self.R, mandatory_nodes=self.mandatory_nodes
            ),
            lambda r, n: regret_2_insertion(
                r, n, self.dist_matrix, self.wastes, self.capacity, R=self.R, mandatory_nodes=self.mandatory_nodes
            ),
        ]

        self.destroy_weights = [1.0] * len(self.destroy_ops)
        self.repair_weights = [1.0] * len(self.repair_ops)

    def _initialize_solve(self, initial_solution: Optional[List[List[int]]]):
        """Initialize the solution and metrics for the solve process."""
        current_routes = initial_solution or self.build_initial_solution()
        best_routes = copy.deepcopy(current_routes)

        # Calculate initial metrics
        best_cost = self.calculate_cost(best_routes)
        best_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in best_routes for node_idx in route)
        best_profit = best_rev - best_cost

        return current_routes, best_routes, best_profit, best_cost

    def _select_and_apply_operators(self, current_routes):
        """Select destroy/repair operators and generate a new solution."""
        d_idx = self.select_operator(self.destroy_weights)
        r_idx = self.select_operator(self.repair_weights)

        destroy_op = self.destroy_ops[d_idx]
        repair_op = self.repair_ops[r_idx]

        n_remove = random.randint(
            self.params.min_removal,
            max(
                self.params.min_removal,
                int(self.n_nodes * self.params.max_removal_pct),
            ),
        )

        partial_routes, removed = destroy_op(copy.deepcopy(current_routes), n_remove)
        new_routes = repair_op(partial_routes, removed)

        return new_routes, d_idx, r_idx

    def _accept_solution(self, current_profit, new_profit, T):
        """Determine whether to accept the new solution based on SA criteria."""
        delta = current_profit - new_profit
        if delta < -1e-6:
            return True
        else:
            prob = math.exp(-delta / T) if T > 0 else 0
            return random.random() < prob

    def _update_weights(self, d_idx, r_idx, score):
        """Update the weights of the used operators."""
        lambda_decay = 0.8
        self.destroy_weights[d_idx] = lambda_decay * self.destroy_weights[d_idx] + (1 - lambda_decay) * max(0.1, score)
        self.repair_weights[r_idx] = lambda_decay * self.repair_weights[r_idx] + (1 - lambda_decay) * max(0.1, score)

    def solve(self, initial_solution: Optional[List[List[int]]] = None) -> Tuple[List[List[int]], float, float]:
        """
        Run the ALNS algorithm.

        Args:
            initial_solution: Optional starting solution. If None, a constructive heuristic is used.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, and total cost.
        """
        (
            current_routes,
            best_routes,
            best_profit,
            best_cost,
        ) = self._initialize_solve(initial_solution)
        current_profit = best_profit

        T = self.params.start_temp
        start_time = time.time()

        for _it in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            new_routes, d_idx, r_idx = self._select_and_apply_operators(current_routes)

            # Calculate new metrics
            new_cost = self.calculate_cost(new_routes)
            new_rev = sum(self.wastes.get(node_idx, 0) * self.R for route in new_routes for node_idx in route)
            new_profit = new_rev - new_cost

            accept = self._accept_solution(current_profit, new_profit, T)
            score = 0

            if accept:
                current_routes = new_routes
                current_profit = new_profit
                if new_profit > best_profit + 1e-6:
                    best_routes = copy.deepcopy(new_routes)
                    best_profit = new_profit
                    best_cost = new_cost
                    score = 3
                else:
                    score = 1

            self._update_weights(d_idx, r_idx, score)
            T *= self.params.cooling_rate

            self._viz_record(
                iteration=_it,
                d_idx=d_idx,
                r_idx=r_idx,
                best_profit=best_profit,
                current_profit=current_profit,
                temperature=T,
                accepted=int(accept),
                score=score,
            )

        return best_routes, best_profit, best_cost

    def select_operator(self, weights: List[float]) -> int:
        """
        Select an operator index based on their weights using roulette wheel selection.

        Args:
            weights: List of operator weights.

        Returns:
            int: Index of the selected operator.
        """
        total = sum(weights)
        r = random.uniform(0, total)
        curr = 0.0
        for i, w in enumerate(weights):
            curr += w
            if curr >= r:
                return i
        return len(weights) - 1

    def calculate_cost(self, routes: List[List[int]]) -> float:
        """
        Calculate the total routing cost for a set of routes.

        Args:
            routes: List of routes.

        Returns:
            float: Total distance * cost multiplier.
        """
        total_dist = 0
        for route in routes:
            if not route:
                continue
            dist = self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                dist += self.dist_matrix[route[i]][route[i + 1]]
            dist += self.dist_matrix[route[-1]][0]
            total_dist += dist
        return total_dist * self.C

    def build_initial_solution(self) -> List[List[int]]:
        """
        Build a basic feasible solution using a greedy constructive heuristic.
        For VRPP, it only includes profitable nodes.

        Returns:
            List[List[int]]: Initial routes.
        """
        nodes = self.nodes[:]
        random.shuffle(nodes)
        routes = []
        curr_route = []
        load = 0.0
        mandatory_set = set(self.mandatory_nodes) if self.mandatory_nodes else set()

        for node in nodes:
            waste = self.wastes.get(node, 0)
            revenue = waste * self.R
            is_mandatory = node in mandatory_set

            # Basic VRPP check for initial solution: is node potentially profitable?
            # (Heuristic: revenue > cost to depot and back)
            if not is_mandatory and revenue < (self.dist_matrix[0][node] + self.dist_matrix[node][0]) * self.C:
                continue

            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            routes.append(curr_route)
        return routes


def run_alns(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args):
    """
    Main ALNS entry point with dispatching to different algorithm variants.

    Args:
        dist_matrix: Distance matrix.
        wastes: Bin wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Dictionary of parameters and config.
        mandatory_nodes: List of mandatory node indices.
        *args: Additional arguments (ignored or passed through).

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, profit, and cost.
    """
    variant = values.get("variant") or values.get("engine") or "custom"

    if variant == "package":
        return run_alns_package(dist_matrix, wastes, capacity, R, C, values)
    elif variant == "ortools":
        return run_alns_ortools(dist_matrix, wastes, capacity, R, C, values)

    # Default: Custom internal ALNS solver
    params = ALNSParams(
        time_limit=values.get("time_limit", 10),
        max_iterations=values.get("max_iterations", 2000),
        start_temp=values.get("start_temp", 100.0),
        cooling_rate=values.get("cooling_rate", 0.995),
        reaction_factor=values.get("reaction_factor", 0.1),
        min_removal=values.get("min_removal", 1),
        max_removal_pct=values.get("max_removal_pct", 0.3),
    )
    solver = ALNSSolver(dist_matrix, wastes, capacity, R, C, params, mandatory_nodes)
    return solver.solve()
