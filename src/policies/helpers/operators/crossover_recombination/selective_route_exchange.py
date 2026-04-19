from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

from .ordered import ordered_crossover

if TYPE_CHECKING:
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import (
        Individual,
    )


def _get_individual_class() -> type:
    """Lazy import to break circular dependency with meta_heuristics.__init__"""
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual

    return Individual


def _select_initial_child_routes(
    p1: Individual,
    rng: random.Random,
) -> Tuple[List[List[int]], Set[int]]:
    """Select exactly one route from Parent 1 to initialize the child."""
    selected_route = rng.choice(p1.routes)
    child_routes = [selected_route[:]]
    child_nodes = set(selected_route)
    return child_routes, child_nodes


def _merge_non_conflicting_p2_routes(
    p2: Individual,
    child_routes: List[List[int]],
    child_nodes: Set[int],
) -> List[int]:
    """Add routes from Parent 2 that don't conflict with child nodes."""
    p2_unassigned = []
    for route in p2.routes:
        route_nodes = set(route)
        if not (route_nodes & child_nodes):  # No overlap
            child_routes.append(route[:])
            child_nodes.update(route)
        else:
            # Collect unassigned P2 nodes for greedy insertion
            for n in route:
                if n not in child_nodes:
                    p2_unassigned.append(n)
    return p2_unassigned


def _insert_missing_nodes_greedy(
    child_routes: List[List[int]],
    child_nodes: Set[int],
    missing: List[int],
    dist_matrix: Optional[np.ndarray],
    wastes: Optional[Dict[int, float]],
    capacity: float,
    R: float,
    C: float,
    rng: random.Random,
) -> None:
    """Greedily insert missing nodes using cheapest insertion cost."""
    if dist_matrix is None or wastes is None:
        return

    rng.shuffle(missing)
    loads = [sum(wastes.get(n, 0) for n in r) for r in child_routes]
    for node in missing:
        node_waste = wastes.get(node, 0.0)
        best_cost = float("inf")
        best_r, best_pos = -1, -1

        for r_idx, route in enumerate(child_routes):
            if loads[r_idx] + node_waste > capacity:
                continue
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                delta = (dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]) * C - node_waste * R
                if delta < best_cost:
                    best_cost = delta
                    best_r, best_pos = r_idx, pos

        if best_r != -1:
            child_routes[best_r].insert(best_pos, node)
            loads[best_r] += node_waste
            child_nodes.add(node)


def _enforce_mandatory_nodes_srex(
    child_routes: List[List[int]],
    mandatory_nodes: List[int],
    wastes: Optional[Dict[int, float]],
    capacity: float,
) -> None:
    """Strictly include all mandatory nodes in the child routes."""
    mandatory_set = set(mandatory_nodes)
    visited_in_routes = {n for route in child_routes for n in route}
    missing_mandatory = mandatory_set - visited_in_routes
    loads = [sum(wastes.get(n, 0.0) for n in r) for r in child_routes] if wastes else [0.0] * len(child_routes)

    for node in missing_mandatory:
        node_waste = wastes.get(node, 0.0) if wastes else 0.0
        best_r = min(range(len(child_routes)), key=lambda i: loads[i], default=None)
        if best_r is not None and (not wastes or loads[best_r] + node_waste <= capacity):
            child_routes[best_r].append(node)
            loads[best_r] += node_waste
        else:
            child_routes.append([node])
            loads.append(node_waste)

    # Pre-Split sanity check: mandatory nodes are in the visited prefix of
    # child_gt, making them more likely to be assigned routes by Split.
    # The definitive enforcement is in LinearSplit.mandatory_nodes.
    visited_prefix = {n for route in child_routes for n in route}
    assert all(n in visited_prefix for n in mandatory_nodes), (
        f"Mandatory nodes missing from routes: {mandatory_set - visited_prefix}"
    )


def selective_route_exchange_crossover(
    p1: Individual,
    p2: Individual,
    rng: Optional[random.Random] = None,
    dist_matrix: Optional[np.ndarray] = None,
    wastes: Optional[Dict[int, float]] = None,
    capacity: float = float("inf"),
    R: float = 1.0,
    C: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Selective Route Exchange Crossover (SREX): Exchanges complete routes.

    Algorithm:
        1. Randomly select exactly one route from Parent 1 (Fix 13).
        2. Add non-conflicting routes from Parent 2.
        3. Collect all nodes that were skipped due to P2 conflicts.
        4. Reinsert missing nodes using cheapest-insertion into physical routes.

    Preserves good route structures from both parents.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.
        dist_matrix: Optional distance matrix for cheapest insertion.
        wastes: Optional node wastes for cheapest insertion.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: List of nodes that MUST be visited.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    # 1. Fallbacks
    if not p1.routes or not p2.routes:
        return ordered_crossover(p1, p2, rng, mandatory_nodes=mandatory_nodes)
    if not p1.giant_tour:
        return _get_individual_class()(p2.giant_tour[:])
    if not p2.giant_tour:
        return _get_individual_class()(p1.giant_tour[:])

    # 2. Select initial P1 routes and merge P2 non-conflicting routes
    child_routes, child_nodes = _select_initial_child_routes(p1, rng)
    p2_unassigned = _merge_non_conflicting_p2_routes(p2, child_routes, child_nodes)

    # 3. Greedy insertion of missing nodes
    missing = [n for n in p1.giant_tour if n not in child_nodes] + p2_unassigned
    # Deduplicate while preserving order
    missing = list(dict.fromkeys(missing))

    _insert_missing_nodes_greedy(child_routes, child_nodes, missing, dist_matrix, wastes, capacity, R, C, rng)

    # 4. Enforce mandatory nodes
    if mandatory_nodes:
        _enforce_mandatory_nodes_srex(child_routes, mandatory_nodes, wastes, capacity)

    # Reconstruct full-length giant tour: route nodes first, then unvisited.
    # Fix 17: Shuffle unvisited suffix to remove Split evaluation bias.
    route_nodes = [node for route in child_routes for node in route]
    visited = set(route_nodes)
    unvisited = [n for n in p1.giant_tour if n not in visited]
    rng.shuffle(unvisited)
    child_gt = route_nodes + unvisited

    return _get_individual_class()(child_gt, expand_pool=p1.expand_pool)
