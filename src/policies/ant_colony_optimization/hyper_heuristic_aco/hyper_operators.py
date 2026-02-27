"""
Hyper-Heuristic Operators Module.

This module provides wrapper functions for local search operators
that can be called as standalone moves in a hyper-heuristic context.
Each operator attempts to improve a solution and returns the modified routes.

Attributes:
    HYPER_OPERATORS (list): List of operator functions.
    OPERATOR_NAMES (list): List of operator names.

Example:
    >>> from logic.src.policies.ant_colony_optimization.hyper_heuristic_aco.hyper_operators import apply_2opt
    >>> improved = apply_2opt(context)
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...operators import (
    kick,
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_relocate,
    move_swap,
    move_swap_star,
    perturb,
)


class HyperOperatorContext:
    """
    Context object that provides the interface expected by move operators.
    Wraps routes and provides helper methods for move evaluation.
    """

    def __init__(
        self,
        routes: List[List[int]],
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize HyperOperatorContext.

        Args:
            routes: Current routes.
            dist_matrix: Distance matrix.
            wastes: Node wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            mandatory_nodes: List of mandatory node indices.
        """
        self.routes = routes
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()
        self.route_loads: List[float] = []
        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.neighbors: Dict[int, List[int]] = {}

        self._build_structures()

    def _build_structures(self):
        """Build node map, route loads, and neighbor lists."""
        self.node_map.clear()
        self.route_loads = []

        for ri, route in enumerate(self.routes):
            load = sum(self.wastes.get(n, 0) for n in route)
            self.route_loads.append(load)
            for pi, node in enumerate(route):
                self.node_map[node] = (ri, pi)

        # Build neighbor lists (k=10 nearest)
        n_nodes = len(self.d)
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

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.wastes.get(x, 0) for x in r)

    def _get_load_cached(self, ri: int) -> float:
        return self.route_loads[ri]

    def _update_map(self, affected_indices: set):
        for ri in affected_indices:
            if ri < len(self.routes):
                for pi, node in enumerate(self.routes[ri]):
                    self.node_map[node] = (ri, pi)
                self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])


# -----------------------------------------------------------------------------
# Wrapper Operators
# -----------------------------------------------------------------------------


def apply_2opt_intra(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply 2-opt intra-route move to a random route segment."""
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = random.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u == r_v and move_2opt_intra(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_3opt_intra(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply 3-opt intra-route move to a random route segment."""
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = random.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u == r_v and move_3opt_intra(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_2opt_star(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply 2-opt* inter-route move."""
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = random.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u != r_v and move_2opt_star(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_swap(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply swap move between two nodes."""
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = random.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if move_swap(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_swap_star(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply SWAP* inter-route move."""
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = random.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u != r_v and move_swap_star(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_relocate(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply relocate move."""
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = random.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if move_relocate(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_perturb(ctx: HyperOperatorContext, k: int = 3) -> bool:
    """Wrapper for perturbation operator."""
    return perturb(ctx, k)


def apply_kick(ctx: HyperOperatorContext, destroy_ratio: float = 0.2) -> bool:
    """Wrapper for kick operator."""
    return kick(ctx, destroy_ratio)


# Registry of available operators
HYPER_OPERATORS: Dict[str, Callable[[HyperOperatorContext], bool]] = {
    "2opt_intra": apply_2opt_intra,
    "3opt_intra": apply_3opt_intra,
    "2opt_star": apply_2opt_star,
    "swap": apply_swap,
    "swap_star": apply_swap_star,
    "relocate": apply_relocate,
    "perturb": apply_perturb,
    "kick": apply_kick,
}

OPERATOR_NAMES = list(HYPER_OPERATORS.keys())
