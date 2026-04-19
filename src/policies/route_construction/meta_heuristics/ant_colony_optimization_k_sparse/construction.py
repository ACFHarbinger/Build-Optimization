"""
Solution Construction Module for K-Sparse ACO.

This module implements the solution construction phase of the MMAS algorithm.
Ants construct solutions using pure roulette-wheel proportional selection
based on pheromone levels and heuristic information.

Attributes:
    None

Example:
    >>> from logic.src.policies.ant_colony_optimization_k_sparse.construction import SolutionConstructor
    >>> constructor = SolutionConstructor(...)
    >>> routes = constructor.construct()

Reference:
    Hale, D. "Investigation of Ant Colony Optimization Implementation
    Strategies For Low-Memory Operating Environments", 2021.
    Uses MMAS_exp transition rule (no local updates, no q0 exploitation).
"""

import random
from typing import Dict, List, Optional, Set

import numpy as np

from .params import KSACOParams
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
        params: KSACOParams,
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
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

    def construct(self) -> List[List[int]]:
        """
        Construct a solution using the MMAS proportional transition rule.

        This method uses pure exploration (roulette-wheel selection) without
        any local pheromone updates or exploitation bias (q0). All pheromone
        updates occur globally after construction.

        Returns:
            List of routes, each a list of node indices.
        """
        unvisited: Set[int] = set(self.nodes)
        mandatory_unvisited = set(self.mandatory_nodes)
        routes: List[List[int]] = []

        # We continue as long as there are mandatory nodes OR if we want to visit more
        # In VRPP mode, if mandatory_unvisited is empty, we only continue if profitable nodes exist.
        while unvisited:
            if not mandatory_unvisited and not self._any_profitable_nodes(unvisited):
                break

            route: List[int] = []
            load = 0.0
            current = 0  # Start from depot

            while unvisited:
                # Find feasible next nodes
                feasible = self._get_feasible_nodes(unvisited, mandatory_unvisited, load, current)

                if not feasible:
                    self._cleanup_unvisited(unvisited, mandatory_unvisited)
                    break

                # Select next node using MMAS proportional rule
                next_node = self._select_next_node(current, sorted(feasible))

                # No local pheromone update in MMAS (removed ACS mechanics)

                route.append(next_node)
                load += self.wastes.get(next_node, 0)
                unvisited.remove(next_node)
                if next_node in mandatory_unvisited:
                    mandatory_unvisited.remove(next_node)
                current = next_node

            if route:
                routes.append(route)

        return routes

    def _any_profitable_nodes(self, unvisited: Set[int]) -> bool:
        """Check if any remaining node is profitable to visit from depot."""
        # In CVRP (not VRPP), all nodes are considered "profitable" to visit until exhausted
        if not self.params.vrpp:
            return True

        for j in sorted(unvisited):
            revenue = self.wastes.get(j, 0) * self.R
            if revenue > (self.dist_matrix[0][j] + self.dist_matrix[j][0]) * self.C:
                return True
        return False

    def _get_feasible_nodes(
        self, unvisited: Set[int], mandatory_unvisited: Set[int], load: float, current: int
    ) -> List[int]:
        """Find nodes that can be added to the current route."""
        feasible = []
        use_profit_check = self.params.vrpp and self.params.profit_aware_operators

        for j in sorted(unvisited):
            if load + self.wastes.get(j, 0) <= self.capacity:
                if j in mandatory_unvisited:
                    feasible.append(j)
                elif use_profit_check:
                    revenue = self.wastes.get(j, 0) * self.R
                    # Skip if immediately unprofitable compared to staying at depot
                    if (
                        revenue
                        > (self.dist_matrix[current][j] + self.dist_matrix[j][0] - self.dist_matrix[current][0])
                        * self.C
                    ):
                        feasible.append(j)
                else:
                    # In CVRP or if profit check is disabled, all capacity-feasible nodes are candidates
                    feasible.append(j)
        return feasible

    def _cleanup_unvisited(self, unvisited: Set[int], mandatory_unvisited: Set[int]) -> None:
        """Remove nodes that can never fit in any route to avoid infinite loops."""
        still_constructible = False
        for j in list(unvisited):
            if self.wastes.get(j, 0) <= self.capacity:
                still_constructible = True
            else:
                unvisited.remove(j)
                if j in mandatory_unvisited:
                    mandatory_unvisited.remove(j)

        if not still_constructible:
            unvisited.clear()

    def _select_next_node(self, current: int, feasible: List[int]) -> int:
        """
        Select next node using MMAS proportional (roulette-wheel) selection.

        This is pure exploration without exploitation bias. The probability
        of selecting node j is proportional to:
            P(j) = [tau(current, j)^alpha] * [eta(current, j)^beta]

        Candidate lists are used to bias selection toward nearby neighbors
        when available, following the k-sparse optimization strategy.

        Args:
            current: Current node index.
            feasible: List of feasible next nodes.

        Returns:
            Selected next node index.
        """
        # Prefer candidates if available (k-nearest neighbors)
        candidates_in_feasible = [j for j in self.candidate_lists.get(current, []) if j in feasible]
        selection_pool = candidates_in_feasible if candidates_in_feasible else feasible

        # Compute selection probabilities
        probs = []
        for j in selection_pool:
            tau = self.pheromone.get(current, j)
            eta = self.eta[current][j]
            probs.append((tau**self.params.alpha) * (eta**self.params.beta))

        total = sum(probs)
        if total <= 0:
            # Fallback to uniform random selection if all probabilities are zero
            return self.random.choice(selection_pool)

        # Roulette-wheel selection
        r = self.random.uniform(0, total)
        cumsum = 0.0
        for idx, p in enumerate(probs):
            cumsum += p
            if cumsum >= r:
                return selection_pool[idx]

        # Fallback (should rarely happen due to floating-point precision)
        return selection_pool[-1]
