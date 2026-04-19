"""
Guided Large Neighborhood Search (G-LNS) for VRPP.

G-LNS bridges the Augmented Cost Function of Guided Local Search (GLS) with
the Ruin-and-Recreate operators of Large Neighborhood Search (LNS). It
augments the objective function with adaptive penalty terms on routing
edge features that appear in local optima. When the inner search stagnates,
the features maximizing a utility function are penalized, and the search
is explicitly biased toward breaking these features using a targeted
'penalized removal' operator (the LNS equivalent of Fast Local Search).

Reference:
    Voudouris, C., & Tsang, E. "Guided Local Search and Its
    Application to the Traveling Salesman Problem", 1999.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    cluster_removal,
    greedy_insertion,
    greedy_profit_insertion,
    penalized_removal,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    worst_removal,
)
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes

from .params import GLSParams


class GLSSolver:
    """
    Guided Large Neighborhood Search (G-LNS) solver for VRPP.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: GLSParams,
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

        # Edge penalty matrix (features = edges)
        n = len(dist_matrix)
        self.penalties = np.zeros((n, n), dtype=np.float64)

        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
            self._llh5,
        ]
        self.base_lambda = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run G-LNS optimisation.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.process_time()

        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # 1. Static Lambda Calculation (Voudouris & Tsang, 1999)
        # Lambda is calculated once based on the initial solution's spatial traversal cost.
        # Scaling by traversal cost rather than net profit decouples the GLS spatial
        # memory mechanism from economic fluctuations and prevents the "zero-lambda" trap.
        initial_cost = self._cost(routes)
        self.base_lambda = self.params.alpha_param * (initial_cost / max(1, self.n_nodes))

        # Flag for Markovian Operator Coupling: bias selection after penalty updates
        penalty_just_updated = False

        for cycle in range(self.params.penalty_cycles):
            if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                break

            # 2. Local Minimum Condition via Stagnation
            # The search continues using the augmented objective until it stabilizes (local minimum).
            stagnation_counter = 0
            while stagnation_counter < self.params.inner_iterations:
                if self.params.time_limit > 0 and time.process_time() - start > self.params.time_limit:
                    break

                # Markovian Operator Coupling:
                # If we just updated penalties, strongly bias selection toward penalized_removal (_llh5)
                # to mimic Fast Local Search (FLS) behavior.
                if penalty_just_updated and self.random.random() < self.params.fls_coupling_prob:
                    llh_idx = 5  # Index of _llh5 (penalized_removal)
                else:
                    llh_idx = self.random.randint(0, len(self._llh_pool) - 1)

                llh = self._llh_pool[llh_idx]

                try:
                    new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                except Exception:
                    stagnation_counter += 1
                    continue

                # Accept if augmented objective improves or stays the same (plateau)
                aug_new = self._augmented_evaluate(new_routes)
                aug_cur = self._augmented_evaluate(routes)

                if aug_new > aug_cur or abs(aug_new - aug_cur) < 1e-9:
                    routes = new_routes
                    real_profit = self._evaluate(routes)

                    # Always check for a global best on any accepted move.
                    # Because Aug = Profit - Lambda * Penalty, a plateau move in
                    # augmented space may represent a strict improvement in real profit.
                    if real_profit > best_profit:
                        best_routes = copy.deepcopy(routes)
                        best_profit = real_profit

                    # Manage stagnation mechanics
                    if aug_new > aug_cur:
                        # To adapt the continuous descent requirement of GLS to the discrete
                        # ruin-and-recreate mechanics of LNS, a "local optimum" is strictly
                        # defined as a state where $k$ consecutive LNS transformations fail to
                        # yield an improvement. Resetting the counter ensures the search
                        # strictly exhausts the current augmented cost basin before triggering
                        # a penalty update.
                        stagnation_counter = 0
                        penalty_just_updated = False
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # At local optimum: penalise all features with maximum utility
            self._update_penalties(routes)
            penalty_just_updated = True

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=cycle,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Penalty management
    # ------------------------------------------------------------------

    def _get_edges(self, routes: List[List[int]]) -> Set[Tuple[int, int]]:
        """Extract all edges from routes (including depot connections)."""
        edges: Set[Tuple[int, int]] = set()
        for route in routes:
            if not route:
                continue
            edges.add((0, route[0]))
            for k in range(len(route) - 1):
                edges.add((route[k], route[k + 1]))
            edges.add((route[-1], 0))
        return edges

    def _update_penalties(self, routes: List[List[int]]) -> None:
        """Penalise all feature(s) with the maximum utility."""
        edges = self._get_edges(routes)
        if not edges:
            return

        # Normalize to unique undirected features to prevent double-penalizing
        # 1-node routes (e.g., [A] -> (0,A) and (A,0) are the same spatial feature).
        undirected_edges = {tuple(sorted(edge)) for edge in edges}

        # 3. Handle Ties in Penalty Updates
        # Identify the maximum utility value among all edges in the current local optimum.
        max_utility = -1.0
        utilities = []

        for i, j in undirected_edges:
            # The feature cost must represent pure objective degradation (spatial
            # traversal cost) to guarantee geometric regularization. It ensures the
            # algorithm does not become trapped traversing globally inefficient edges
            # simply because they connect locally profitable nodes.
            cost_ij = self.dist_matrix[i][j]

            utility = cost_ij / (1.0 + self.penalties[i][j])
            utilities.append(((i, j), utility))
            if utility > max_utility:
                max_utility = utility

        # Penalize ALL edges that share the maximum utility (within epsilon)
        if max_utility > -1.0:
            for edge, utility in utilities:
                if utility >= max_utility - 1e-6:
                    # Treat edges as undirected features to prevent trivial escape
                    # via route inversion.
                    self.penalties[edge[0]][edge[1]] += 1.0
                    self.penalties[edge[1]][edge[0]] += 1.0

    def _augmented_evaluate(self, routes: List[List[int]]) -> float:
        """Evaluate with penalty-augmented objective using static lambda."""
        real = self._evaluate(routes)
        penalty = 0.0

        for route in routes:
            if not route:
                continue
            penalty += self.penalties[0][route[0]]
            for k in range(len(route) - 1):
                penalty += self.penalties[route[k]][route[k + 1]]
            penalty += self.penalties[route[-1]][0]

        # Profit' = Profit - (Weight * Base_Lambda * PenaltySum)
        # Note: self.params.lambda_param acts as the global scaling weight
        return real - self.params.lambda_param * self.base_lambda * penalty

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = random_removal(routes, n, self.random)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = random_removal(routes, n, self.random)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = cluster_removal(routes, n, self.dist_matrix, self.nodes)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = worst_removal(routes, n, self.dist_matrix)
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        if self.params.profit_aware_operators:
            partial, removed = random_removal(routes, n, self.random)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
            )
        else:
            partial, removed = random_removal(routes, n, self.random)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=self.params.vrpp,
            )

    def _llh5(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """Penalized Removal (Targeted Destruction): The G-LNS equivalent of FLS."""
        partial, removed = penalized_removal(routes, n, self.penalties, rng=self.random)  # type: ignore[arg-type]
        if self.params.profit_aware_operators:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                self.mandatory_nodes,
                self.params.vrpp,
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_solution(self) -> List[List[int]]:
        routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
        )
        return routes

    def _evaluate(self, routes: List[List[int]]) -> float:
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
