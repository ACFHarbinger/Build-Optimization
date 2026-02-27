"""
K-Sparse ACO Solver Module.

This module implements the main loop of the K-Sparse Ant Colony Optimization
algorithm. It manages the ant colony, pheromone updates (local and global),
and coordinates the search process.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization.k_sparse_aco.solver import KSparseACOSolver
    >>> solver = KSparseACOSolver(dist_matrix, wastes, ...)
    >>> result = solver.solve()
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ...local_search.local_search_aco import ACOLocalSearch
from .construction import SolutionConstructor
from .params import ACOParams
from .pheromones import SparsePheromoneTau


class KSparseACOSolver(PolicyVizMixin):
    """
    K-Sparse Ant Colony System solver for CVRP/VRPP.

    Implements ACS with sparse pheromone storage for memory efficiency
    and fast computation on large problem instances.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: ACOParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the K-Sparse ACO solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Dictionary of node wastes {node_idx: waste}.
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            params: ACO hyperparameters.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(dist_matrix)
        self.nodes = list(range(1, self.n_nodes))  # Exclude depot (0)

        # Precompute heuristic values (eta = 1/distance)
        self.eta = np.zeros_like(dist_matrix, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)

        # Compute initial pheromone based on nearest neighbor heuristic
        if params.tau_0 is None:
            nn_cost = self._nearest_neighbor_cost()
            self.tau_0 = 1.0 / (self.n_nodes * nn_cost) if nn_cost > 0 else params.tau_max
        else:
            self.tau_0 = params.tau_0

        # Initialize sparse pheromone matrix
        self.pheromone = SparsePheromoneTau(
            self.n_nodes,
            params.k_sparse,
            self.tau_0,
            params.tau_min,
            params.tau_max,
        )

        # Initialize Local Search
        self.ls = ACOLocalSearch(dist_matrix, wastes, capacity, R, C, params)

        # Build candidate lists (k-nearest neighbors for each node)
        self.candidate_lists = self._build_candidate_lists()

        # Initialize Constructor
        self.constructor = SolutionConstructor(
            dist_matrix,
            wastes,
            capacity,
            self.pheromone,
            self.eta,
            self.candidate_lists,
            self.nodes,
            params,
            self.tau_0,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _nearest_neighbor_cost(self) -> float:
        """Compute cost of nearest neighbor tour for tau_0 initialization."""
        visited = set([0])
        current = 0
        cost = 0.0
        for _ in range(len(self.nodes)):
            best_next = None
            best_dist = float("inf")
            for node in self.nodes:
                if node not in visited:
                    d = self.dist_matrix[current][node]
                    if d < best_dist:
                        best_dist = d
                        best_next = node
            if best_next is not None:
                cost += best_dist
                visited.add(best_next)
                current = best_next
        cost += self.dist_matrix[current][0]  # Return to depot
        return cost

    def _build_candidate_lists(self) -> Dict[int, List[int]]:
        """Build k-nearest neighbor candidate lists for each node."""
        candidates: Dict[int, List[int]] = {}
        k = min(self.params.k_sparse, len(self.nodes))

        for i in range(self.n_nodes):
            # Get distances to all other nodes
            distances = [(self.dist_matrix[i][j], j) for j in range(self.n_nodes) if j != i]
            distances.sort()
            candidates[i] = [j for _, j in distances[:k]]

        return candidates

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the K-Sparse ACO algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: (best_routes, profit, cost)
        """
        best_routes: List[List[int]] = []
        best_cost = float("inf")
        start_time = time.time()

        for _iteration in range(self.params.max_iterations):
            if time.time() - start_time > self.params.time_limit:
                break

            iteration_best_routes: List[List[int]] = []
            iteration_best_cost = float("inf")

            # Each ant constructs a solution
            for _ in range(self.params.n_ants):
                # Use delegated constructor
                routes = self.constructor.construct()

                # Optional local search
                if self.params.local_search:
                    routes = self.ls.optimize(routes)

                cost = self._calculate_cost(routes)

                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_routes = routes

            # Update global best
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_routes = iteration_best_routes

            # Global pheromone update
            self._global_pheromone_update(best_routes, best_cost)

            _tau_vals = [v for nbrs in self.pheromone._pheromone.values() for v in nbrs.values()]
            self._viz_record(
                iteration=_iteration,
                best_cost=best_cost,
                iter_best_cost=iteration_best_cost,
                tau_mean=float(sum(_tau_vals) / len(_tau_vals)) if _tau_vals else self.pheromone.tau_0,
                tau_max=float(max(_tau_vals)) if _tau_vals else self.pheromone.tau_0,
            )

        # Calculate profit
        collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in best_routes for node in route)
        profit = collected_revenue - best_cost * self.C

        return best_routes, profit, best_cost

    def _global_pheromone_update(self, best_routes: List[List[int]], best_cost: float) -> None:
        """
        Apply ACS global pheromone update on best-so-far solution.

        Only edges in the best solution receive pheromone deposit.
        """
        if not best_routes or best_cost <= 0:
            return

        # Evaporate all pheromones
        self.pheromone.evaporate_all(self.params.rho)

        # Deposit on best solution edges
        delta = self.params.elitist_weight / best_cost

        for route in best_routes:
            if not route:
                continue

            # Depot to first node
            self.pheromone.update_edge(0, route[0], delta, evaporate=False)

            # Route edges
            for k in range(len(route) - 1):
                self.pheromone.update_edge(route[k], route[k + 1], delta, evaporate=False)

            # Last node back to depot
            self.pheromone.update_edge(route[-1], 0, delta, evaporate=False)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total
