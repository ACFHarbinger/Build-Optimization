"""
Hyper-Heuristic Operators Module.

This module provides wrapper functions for local search operators
that can be called as standalone moves in a hyper-heuristic context.
Each operator attempts to improve a solution and returns the modified routes.

Attributes:
    HYPER_OPERATORS (list): List of operator functions.
    OPERATOR_NAMES (list): List of operator names.

Example:
    >>> from logic.src.policies.ant_colony_optimization_hyper_heuristic.hyper_operators import apply_2opt
    >>> improved = apply_2opt(context)
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    kick,
    kick_profit,
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_relocate,
    move_swap,
    move_swap_star,
    perturb,
    random_removal,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
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
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        mandatory_nodes: Optional[List[int]] = None,
        rng: Optional[random.Random] = None,
        profit_aware_operators: bool = False,
        vrpp: bool = True,
    ):
        """
        Initialize HyperOperatorContext.

        Args:
            routes: Current routes.
            dist_matrix: Distance matrix.
            waste: Node wastes.
            capacity: Vehicle capacity.
            R: Revenue multiplier.
            C: Cost multiplier.
            mandatory_nodes: List of mandatory node indices.
            rng: Random number generator.
            profit_aware_operators: Whether to use profit-aware heuristics.
            vrpp: Whether to consider all unvisited nodes in insertion.
        """
        self.routes = routes
        self.d = dist_matrix
        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()
        self.profit_aware_operators = profit_aware_operators
        self.vrpp = vrpp
        self.route_loads: List[float] = []
        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.neighbors: Dict[int, List[int]] = {}
        self.rng = rng or random.Random()

        # Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        # Maps: node_id -> route_idx -> [(cost_delta, position), ...]
        self.top_insertions: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}

        self._build_structures()

    def _build_structures(self):
        """Build node map, route loads, and neighbor lists."""
        self.node_map.clear()
        self.route_loads = []

        for ri, route in enumerate(self.routes):
            load = sum(self.waste.get(n, 0) for n in route)
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

        # Compute Top-3 Insertion Cache for SWAP* operator
        self._compute_top_insertions()

    def _calc_load_fresh(self, r: List[int]) -> float:
        return sum(self.waste.get(x, 0) for x in r)

    def _get_load_cached(self, ri: int) -> float:
        return self.route_loads[ri]

    def _update_map(self, affected_indices: set):
        for ri in sorted(affected_indices):
            if ri < len(self.routes):
                for pi, node in enumerate(self.routes[ri]):
                    self.node_map[node] = (ri, pi)
                self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
                # Update top insertions for affected route
                self._compute_top_insertions(route_idx=ri)

    def _compute_top_insertions(self, route_idx: Optional[int] = None):
        """
        Compute or update the Top-3 Insertion Cache for O(1) SWAP* evaluation.

        Following Vidal et al. (2022), this cache stores the 3 best insertion positions
        for each node into each route, enabling constant-time insertion cost evaluation.

        Args:
            route_idx: If provided, update cache only for this route. If None, initialize for all routes.
        """
        routes_to_update = [route_idx] if route_idx is not None else range(len(self.routes))

        for r_idx in routes_to_update:
            route = self.routes[r_idx]
            if not route:
                continue

            # For each node that could potentially be inserted into this route
            for node in self.neighbors:
                # Skip nodes already in this route
                if node in route:
                    continue

                # Initialize cache structure if needed
                if node not in self.top_insertions:
                    self.top_insertions[node] = {}

                # Calculate insertion costs for all positions
                insertion_costs = []
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = self.d[prev, node] + self.d[node, nxt] - self.d[prev, nxt]
                    insertion_costs.append((delta, pos))

                # Keep top 3 (lowest cost) insertions
                insertion_costs.sort(key=lambda x: x[0])
                self.top_insertions[node][r_idx] = insertion_costs[:3]


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
        u = ctx.rng.choice(nodes)
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
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u == r_v and move_3opt_intra(ctx, u, v, r_u, p_u, r_v, p_v, ctx.rng):
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
        u = ctx.rng.choice(nodes)
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
        u = ctx.rng.choice(nodes)
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
        u = ctx.rng.choice(nodes)
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
        u = ctx.rng.choice(nodes)
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
    if ctx.profit_aware_operators:
        return kick_profit(ctx, destroy_ratio, bias=2.0, rng=ctx.rng)
    return kick(ctx, destroy_ratio, rng=ctx.rng)


def apply_shaw_removal(ctx: HyperOperatorContext, n: Optional[int] = None) -> bool:
    """Wrapper for shaw removal followed by greedy insertion."""
    n_remove = n if n is not None else max(1, len(ctx.node_map) // 5)
    try:
        if ctx.profit_aware_operators:
            partial, removed = shaw_profit_removal(ctx.routes, n_remove, ctx.d, ctx.waste, ctx.R, ctx.C, rng=ctx.rng)
        else:
            partial, removed = shaw_removal(ctx.routes, n_remove, ctx.d, rng=ctx.rng)

        if ctx.profit_aware_operators:
            ctx.routes = greedy_profit_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                R=ctx.R,
                C=ctx.C,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=ctx.vrpp,
            )
        else:
            ctx.routes = greedy_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=ctx.vrpp,
            )
        ctx._build_structures()
        return True
    except Exception:
        return False


def apply_string_removal(ctx: HyperOperatorContext, n: Optional[int] = None) -> bool:
    """Wrapper for string removal followed by greedy insertion."""
    n_remove = n if n is not None else max(1, len(ctx.node_map) // 5)
    try:
        partial, removed = string_removal(ctx.routes, n_remove, ctx.d, rng=ctx.rng)
        if ctx.profit_aware_operators:
            ctx.routes = greedy_profit_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                R=ctx.R,
                C=ctx.C,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=ctx.vrpp,
            )
        else:
            ctx.routes = greedy_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=ctx.vrpp,
            )
        ctx._build_structures()
        return True
    except Exception:
        return False


def apply_random_removal(ctx: HyperOperatorContext, n: Optional[int] = None) -> bool:
    """Wrapper for random removal followed by greedy insertion."""
    n_remove = n if n is not None else max(1, len(ctx.node_map) // 5)
    try:
        partial, removed = random_removal(ctx.routes, n_remove, rng=ctx.rng)
        if ctx.profit_aware_operators:
            ctx.routes = greedy_profit_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                R=ctx.R,
                C=ctx.C,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=ctx.vrpp,
            )
        else:
            ctx.routes = greedy_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=ctx.vrpp,
            )
        ctx._build_structures()
        return True
    except Exception:
        return False


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
    "shaw_removal": apply_shaw_removal,
    "string_removal": apply_string_removal,
    "random_removal": apply_random_removal,
}

OPERATOR_NAMES = list(HYPER_OPERATORS.keys())
