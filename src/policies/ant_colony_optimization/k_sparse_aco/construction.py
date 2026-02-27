"""
Solution Construction Module for K-Sparse ACO.

This module implements the solution construction phase of the ACS algorithm.
Ants construct solutions by probabilistically selecting edges based on pheromone
levels and heuristic information.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization.k_sparse_aco.construction import SolutionConstructor
    >>> constructor = SolutionConstructor(...)
    >>> routes = constructor.construct()
"""

import random
from typing import Dict, List, Optional, Set

import numpy as np

from .params import ACOParams
from .pheromones import SparsePheromoneTau


class SolutionConstructor:
    """
    Constructs a single solution (route set) for an ant using the
    k-sparse pheromone matrix and heuristic values.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        pheromone: SparsePheromoneTau,
        eta: np.ndarray,
        candidate_lists: Dict[int, List[int]],
        nodes: List[int],
        params: ACOParams,
        tau_0: float,
        R: float = 0.0,
        C: float = 1.0,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize Solution Constructor.

        Args:
            dist_matrix: Distance matrix between nodes.
            wastes: Dictionary of node wastes.
            capacity: Vehicle capacity.
            pheromone: Sparse pheromone matrix.
            eta: Heuristic information matrix (inverse of distances).
            candidate_lists: Precomputed candidate lists for each node.
            nodes: List of all nodes (excluding depot).
            params: ACO parameters.
            tau_0: Initial pheromone value.
            R: Revenue multiplier.
            C: Cost multiplier.
            mandatory_nodes: List of mandatory node indices.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.pheromone = pheromone
        self.eta = eta
        self.candidate_lists = candidate_lists
        self.nodes = nodes
        self.params = params
        self.tau_0 = tau_0
        self.R = R
        self.C = C
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()

    def construct(self) -> List[List[int]]:
        """
        Construct a solution using the ACS state transition rule.

        Returns:
            List of routes, each a list of node indices.
        """
        unvisited: Set[int] = set(self.nodes)
        mandatory_unvisited = set(self.mandatory_nodes)
        routes: List[List[int]] = []

        # We continue as long as there are mandatory nodes OR if we want to visit more
        # In VRPP mode, if mandatory_unvisited is empty, we only continue if profitable nodes exist.
        while unvisited:
            if not mandatory_unvisited:
                # Optional: break if no profitable/useful nodes left in unvisited
                # For ACS, we usually continue until a termination condition.
                # Here we check if any remaining node is profitable.
                has_profitable = False
                for j in unvisited:
                    revenue = self.wastes.get(j, 0) * self.R
                    # Heuristic check: is revenue > cost to depot and back?
                    if revenue > (self.dist_matrix[0][j] + self.dist_matrix[j][0]) * self.C:
                        has_profitable = True
                        break
                if not has_profitable:
                    break

            route: List[int] = []
            load = 0.0
            current = 0  # Start from depot

            while unvisited:
                # Find feasible next nodes
                feasible = []
                for j in unvisited:
                    if load + self.wastes.get(j, 0) <= self.capacity:
                        # VRPP Logic: Only consider j if mandatory or potentially profitable
                        if j in mandatory_unvisited:
                            feasible.append(j)
                        else:
                            revenue = self.wastes.get(j, 0) * self.R
                            # Skip if immediately unprofitable compared to staying at depot
                            # (Very conservative check)
                            if (
                                revenue
                                > (self.dist_matrix[current][j] + self.dist_matrix[j][0] - self.dist_matrix[current][0])
                                * self.C
                            ):
                                feasible.append(j)

                if not feasible:
                    break

                # Select next node
                next_node = self._select_next_node(current, feasible)

                # Local pheromone update (ACS rule)
                self._local_pheromone_update(current, next_node)

                route.append(next_node)
                load += self.wastes.get(next_node, 0)
                unvisited.remove(next_node)
                if next_node in mandatory_unvisited:
                    mandatory_unvisited.remove(next_node)
                current = next_node

            if route:
                routes.append(route)

        return routes

    def _select_next_node(self, current: int, feasible: List[int]) -> int:
        """
        Select next node using pseudo-random proportional rule.

        With probability q0, select best node (exploitation).
        Otherwise, use roulette wheel selection (exploration).
        """
        if random.random() < self.params.q0:
            # Exploitation: select best
            best_node = max(
                feasible,
                key=lambda j: (
                    self.pheromone.get(current, j) ** self.params.alpha * self.eta[current][j] ** self.params.beta
                ),
            )
            return best_node
        else:
            # Exploration: proportional selection
            # Prefer candidates if available
            candidates_in_feasible = [j for j in self.candidate_lists.get(current, []) if j in feasible]
            selection_pool = candidates_in_feasible if candidates_in_feasible else feasible

            probs = []
            for j in selection_pool:
                tau = self.pheromone.get(current, j)
                eta = self.eta[current][j]
                probs.append((tau**self.params.alpha) * (eta**self.params.beta))

            total = sum(probs)
            if total <= 0:
                return random.choice(selection_pool)

            r = random.uniform(0, total)
            cumsum = 0.0
            for idx, p in enumerate(probs):
                cumsum += p
                if cumsum >= r:
                    return selection_pool[idx]

            return selection_pool[-1]

    def _local_pheromone_update(self, i: int, j: int) -> None:
        """
        Apply ACS local pheromone update rule.

        tau(i,j) = (1 - rho) * tau(i,j) + rho * tau_0
        """
        rho = self.params.rho
        current = self.pheromone.get(i, j)
        new_value = (1 - rho) * current + rho * self.tau_0
        self.pheromone.set(i, j, new_value)
