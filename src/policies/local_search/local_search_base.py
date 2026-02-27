"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

Attributes:
    None

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyVizMixin

from ..operators import (
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_relocate,
    move_swap,
    move_swap_star,
)


class LocalSearch(PolicyVizMixin, ABC):
    """
    Abstract base class for Local Search algorithms.
    Provides common infrastructure for neighbor lists and move operators.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
    ):
        """
        Initialize Local Search base class.

        Args:
            dist_matrix: The distance matrix between nodes.
            waste: A dictionary mapping node IDs to their waste amounts.
            capacity: The maximum capacity of a vehicle.
            R: A parameter, likely related to route cost or penalty.
            C: A parameter, likely related to route cost or penalty.
            params: An object containing additional parameters for the local search,
                    such as time_limit.
        """
        self.d = np.array(dist_matrix)
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params

        # Common initialization for neighbors (used by all LS)
        n_nodes = len(dist_matrix)
        self.neighbors = {}
        for i in range(1, n_nodes):
            row = self.d[i]
            order = np.argsort(row)
            cands = []
            for c in order:
                if c not in (i, 0):
                    cands.append(c)
                    if len(cands) >= 10:
                        break
            self.neighbors[i] = cands

        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """
        Optimize the given solution.
        """
        pass

    def _optimize_internal(self):
        """
        Core local search loop. Assumes self.routes is populated.
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]

        # Initialize node map
        self.node_map.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        improved = True
        limit = 500  # Safety cap
        it = 0
        t_start = time.time()

        while improved and it < limit:
            improved = False
            it += 1
            if it % 50 == 0 and (time.time() - t_start > self.params.time_limit):
                break

            nodes = [n for n in self.neighbors if n in self.node_map]
            random.shuffle(nodes)

            for u in nodes:
                if self._process_node(u):
                    improved = True
                    break

            self._viz_record(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.waste.get(x, 0) for x in r)

    def _process_node(self, u: int) -> bool:
        u_loc = self.node_map.get(u)
        if not u_loc:
            return False
        r_u, p_u = u_loc

        for v in self.neighbors[u]:
            v_loc = self.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc

            if self._move_relocate(u, v, r_u, p_u, r_v, p_v):
                return True
            if self._move_swap(u, v, r_u, p_u, r_v, p_v):
                return True
            if r_u != r_v:
                if self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_swap_star(u, v, r_u, p_u, r_v, p_v):
                    return True
            else:
                if self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True
                if self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v):
                    return True

        return False

    def _update_map(self, affected_indices: Set[int]):
        for ri in affected_indices:
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])

    def _get_load_cached(self, ri: int) -> float:
        return self.route_loads[ri]

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)
