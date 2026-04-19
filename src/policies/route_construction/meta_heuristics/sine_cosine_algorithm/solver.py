"""
Sine Cosine Algorithm (SCA) for VRPP.

Positions are real-valued vectors updated each iteration by trigonometric
wave functions.  Exploration / exploitation balance is enforced by the
parameter `a` which decays linearly from `a_max` to 0 over the run.  The
final continuous vector is binarised via a sigmoid transfer function and
decoded to a discrete routing solution via the Largest Rank Value (LRV) rule.

Reference:
    Mirjalili, S. "SCA: A Sine Cosine Algorithm for solving
    optimization problems.", 2016, Knowledge-Based Systems.
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch
from logic.src.policies.helpers.operators.recreate_repair.greedy import (
    greedy_insertion,
    greedy_profit_insertion,
)
from logic.src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams

from .params import SCAParams


class SCASolver:
    """
    Sine Cosine Algorithm solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SCAParams,
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        self.np_rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()

        # Initialize Local Search for elite learning
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run SCA and return the best solution found.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()
        T = self.params.max_iterations

        # Initialise population in continuous space
        X = self.np_rng.uniform(-1.0, 1.0, (self.params.pop_size, self.n_nodes))
        routes_pop = [self._decode(x) for x in X]
        profits = [self._evaluate(r) for r in routes_pop]

        best_idx = int(np.argmax(profits))
        X_best = X[best_idx].copy()
        best_routes = copy.deepcopy(routes_pop[best_idx])
        best_profit = profits[best_idx]
        best_cost = self._cost(best_routes)

        for t in range(T):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # Control parameter a decays linearly from a_max → 0 (Mirjalili 2016, Eq 3.4)
            # r1 = a - t * (a / T)
            r1 = self.params.a_max - t * (self.params.a_max / T)

            for i in range(self.params.pop_size):
                # r2 in [0, 2*pi] determines how far the movement is (Eq 3.1 & 3.2)
                r2 = self.random.uniform(0, 2 * math.pi)
                # r3 in [0, 2] provides random weighting for the destination (X_best)
                # (Eq 3.1 & 3.2) - stochastically emphasizes (r3 > 1) or
                # de-emphasizes (r3 < 1) the effect of the destination.
                r3 = self.random.uniform(0, 2)
                # r4 in [0, 1] switches between sine and cosine (Eq 3.3)
                r4 = self.random.random()

                # Core SCA update equations (Mirjalili 2016, Eq 3.1 & 3.2)
                if r4 < 0.5:
                    # Sine phase (Eq 3.1)
                    X[i] = X[i] + r1 * math.sin(r2) * np.abs(r3 * X_best - X[i])
                else:
                    # Cosine phase (Eq 3.2)
                    X[i] = X[i] + r1 * math.cos(r2) * np.abs(r3 * X_best - X[i])

                # Clipping to search space [-1, 1]
                X[i] = np.clip(X[i], -1.0, 1.0)

                # Decode and evaluate
                routes_pop[i] = self._decode(X[i])
                profits[i] = self._evaluate(routes_pop[i])

                if profits[i] > best_profit:
                    X_best = X[i].copy()
                    best_routes = copy.deepcopy(routes_pop[i])
                    best_profit = profits[i]
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=t,
                best_profit=best_profit,
                best_cost=best_cost,
                r1=r1,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode(self, x: np.ndarray) -> List[List[int]]:
        """
        Decode a continuous position vector to a discrete routing solution.
        """
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        mandatory_set = set(self.mandatory_nodes)
        selected_nodes: List[int] = []

        for node in self.nodes:
            if node in mandatory_set:
                selected_nodes.append(node)

        optional_candidates = []
        for idx, node in enumerate(self.nodes):
            if node not in mandatory_set:
                optional_candidates.append((sigmoid[idx], node))

        optional_candidates.sort(key=lambda item: item[0], reverse=True)

        load = sum(self.wastes.get(n, 0.0) for n in selected_nodes)
        for s_val, node in optional_candidates:
            w = self.wastes.get(node, 0.0)
            if s_val > 0.5 and load + w <= self.capacity:
                selected_nodes.append(node)
                load += w

        if not selected_nodes and optional_candidates:
            _, best_node = optional_candidates[0]
            selected_nodes.append(best_node)

        if not selected_nodes:
            return []

        if self.params.profit_aware_operators:
            routes = greedy_profit_insertion(
                [[]],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )
        else:
            routes = greedy_insertion(
                [[]],
                selected_nodes,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

        # Apply local search refinement
        if self.params.local_search_iterations > 0:
            routes = self.ls.optimize(routes)

        return [r for r in routes if r]  # Clean up any empty routes

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
