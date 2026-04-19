import copy
from typing import Dict, List

import numpy as np


class ALNSPerturbationContext:
    """
    Context object required by perturbation operators.

    This class serves as a state-carrier for the various perturbation
    and destruction heuristics, providing a unified interface for
    route data, waste demands, and distance matrices.
    """

    def __init__(self, routes: List[List[int]], dist_matrix: np.ndarray, wastes: Dict[int, float], capacity: float):
        """
        Initialize the perturbation context.

        Args:
            routes (List[List[int]]): Current solution routes.
            dist_matrix (np.ndarray): NxN distance matrix.
            wastes (Dict[int, float]): Dictionary of node waste demands.
            capacity (float): Maximum vehicle capacity.
        """
        self.routes = copy.deepcopy(routes)
        self.dist_matrix = dist_matrix
        self.d = dist_matrix
        self.waste = wastes
        self.Q = capacity
        self._build_structures()

    def _build_structures(self) -> None:
        """Construct auxiliary structures like node_map for efficient lookup."""
        self.node_map = {}
        for r_idx, route in enumerate(self.routes):
            for pos, node in enumerate(route):
                self.node_map[node] = (r_idx, pos)

    def _update_map(self, changed_routes: set) -> None:
        """
        Rebuild internal maps after modifications.

        Args:
            changed_routes (set): Indices of routes that were modified (for efficiency).
        """
        self._build_structures()

    def _get_load_cached(self, ri: int) -> float:
        """
        Calculate total waste load of a specific route.

        Args:
            ri (int): Index of the route.

        Returns:
            float: Sum of waste for all nodes in the route.
        """
        return sum(self.waste.get(n, 0) for n in self.routes[ri])
