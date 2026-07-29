from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import (
        Individual,
    )


def _get_individual_class() -> type:
    """Lazy import of Individual to break the circular import."""
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual

    return Individual


def _dfs_iterative(
    start: int,
    adj: Dict[int, List[int]],
    visited: Set[int],
) -> List[int]:
    """Iterative Depth First Search to avoid RecursionError (Fix 15)."""
    component: List[int] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        component.append(node)
        stack.extend(adj[node])
    return component


def get_components(adj: Dict[int, List[int]], all_nodes: Set[int]) -> List[List[int]]:
    """Find connected components using iterative DFS (Fix 15)."""
    visited: Set[int] = {0}  # Mark depot as visited
    components: List[List[int]] = []
    for node in all_nodes:
        if node not in visited and node != 0:
            comp = _dfs_iterative(node, adj, visited)
            if comp:
                components.append(comp)
    return components


def generalized_partition_crossover(
    p1: Individual,
    p2: Individual,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
    wastes: Optional[Dict[int, float]] = None,
    capacity: float = float("inf"),
) -> Individual:
    """
    Generalized Partition Crossover (GPX): Graph-based recombination.

    Redirects to route_profit_gpx_crossover if routes are available,
    otherwise falls back to ordered_crossover (Fix 16).
    """
    if rng is None:
        rng = random.Random()

    if p1.routes and p2.routes:
        return route_profit_gpx_crossover(
            p1,
            p2,
            dist_matrix=None,
            wastes=wastes if wastes is not None else {},
            capacity=capacity,
            rng=rng,
            mandatory_nodes=mandatory_nodes,
        )

    from .ordered import ordered_crossover

    return ordered_crossover(p1, p2, rng=rng, mandatory_nodes=mandatory_nodes)


def _get_physical_edges(routes: List[List[int]]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for route in routes:
        if not route:
            continue
        edges.add((0, route[0]))
        for i in range(len(route) - 1):
            edges.add((route[i], route[i + 1]))
        edges.add((route[-1], 0))
    return edges


def _get_components_from_uncommon_edges(
    p1_gt: List[int],
    p2_gt: List[int],
    uncommon_edges: Set[Tuple[int, int]],
) -> List[List[int]]:
    """Build adjacency from uncommon edges and find components."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in uncommon_edges:
        if u != 0 and v != 0:
            adj[u].append(v)
            adj[v].append(u)

    all_nodes = set(p1_gt) | set(p2_gt)
    return get_components(adj, all_nodes)


def _inherit_components(
    components: List[List[int]],
    p1: Individual,
    p2: Individual,
    wastes: Dict[int, float],
    capacity: float,
    rng: random.Random,
) -> Tuple[List[List[int]], Set[int]]:
    """Selectively inherit components as routes if they fit capacity."""
    child_routes: List[List[int]] = []
    child_nodes: Set[int] = set()
    skip_capacity_check = not wastes and capacity < float("inf")

    for component in components:
        if not skip_capacity_check:
            comp_load = sum(wastes.get(n, 0.0) for n in component)
            if comp_load > capacity:
                continue

        # Fix 8: Prefer physical route order; fall back to giant tour order.
        if rng.random() < 0.5:
            source_routes, fallback_tour = p1.routes, p1.giant_tour
        else:
            source_routes, fallback_tour = p2.routes, p2.giant_tour

        comp_set = set(component)
        ordered_comp: List[int] = []
        ordered_comp_set: Set[int] = set()

        if source_routes:
            for route in source_routes:
                for n in route:
                    if n in comp_set and n not in ordered_comp_set:
                        ordered_comp.append(n)
                        ordered_comp_set.add(n)

        # Catch any component nodes not in any route
        for n in fallback_tour:
            if n in comp_set and n not in ordered_comp_set:
                ordered_comp.append(n)
                ordered_comp_set.add(n)

        if ordered_comp:
            child_routes.append(ordered_comp)
            child_nodes.update(ordered_comp)
    return child_routes, child_nodes


def _greedy_pack_pool(
    pool: List[int],
    child_routes: List[List[int]],
    child_nodes: Set[int],
    loads: List[float],
    dist_matrix: Optional[np.ndarray],
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_set: Set[int],
) -> None:
    """Greedily pack remaining nodes into routes based on profit."""
    for node in pool:
        node_waste = wastes.get(node, 0.0) if wastes else 0.0
        revenue = node_waste * R

        if dist_matrix is not None and wastes:
            best_profit = -float("inf")
            best_r_idx = -1
            best_pos = -1

            for r_idx, route in enumerate(child_routes):
                if loads[r_idx] + node_waste > capacity:
                    continue
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    profit = revenue - (cost * C)

                    if profit > best_profit and profit >= -1e-4:
                        best_profit = profit
                        best_r_idx = r_idx
                        best_pos = pos

            new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
            new_profit = revenue - (new_cost * C)

            # STANDALONE break-even seed hurdle
            is_mandatory = node in mandatory_set
            if (new_profit >= 0.0 or is_mandatory) and new_profit > best_profit:
                child_routes.append([node])
                loads.append(node_waste)
                child_nodes.add(node)
            elif best_r_idx != -1:
                child_routes[best_r_idx].insert(best_pos, node)
                loads[best_r_idx] += node_waste
                child_nodes.add(node)
        else:
            # Fix 7: Distance-free fallback: insert into route with most remaining
            # capacity, only if the node fits
            best_r = -1
            best_remaining = -1.0
            for r_idx in range(len(child_routes)):
                remaining_cap = capacity - loads[r_idx]
                if remaining_cap >= node_waste and remaining_cap > best_remaining:
                    best_remaining = remaining_cap
                    best_r = r_idx
            if best_r != -1:
                child_routes[best_r].append(node)
                loads[best_r] += node_waste
                child_nodes.add(node)
            # Uninserted nodes remain unvisited (correct VRPP behaviour)


def _enforce_mandatory_nodes(
    child_routes: List[List[int]],
    mandatory_nodes: List[int],
    wastes: Dict[int, float],
    capacity: float,
    loads: List[float],
) -> None:
    """Ensure all mandatory nodes are strictly included in some route."""
    mandatory_set = set(mandatory_nodes)
    visited_in_routes = {n for route in child_routes for n in route}
    missing_mandatory = mandatory_set - visited_in_routes

    for node in missing_mandatory:
        node_waste = wastes.get(node, 0.0)
        # Fix 9: Use maximum remaining capacity for mandatory node enforcement.
        best_r = max(
            range(len(child_routes)),
            key=lambda i: capacity - loads[i],
            default=None,
        )
        if best_r is not None and loads[best_r] + node_waste <= capacity:
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


def route_profit_gpx_crossover(
    p1: Individual,
    p2: Individual,
    dist_matrix: Optional[np.ndarray],
    wastes: Dict[int, float],
    capacity: float,
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Route-based Profit Generalized Partition Crossover (RP-GPX).
    Adapts the classical GPX for VRPP by operating on decoded routes
    rather than the giant tour.
    """
    if rng is None:
        rng = random.Random()

    if not p1.routes or not p2.routes:
        return _get_individual_class()(p1.giant_tour[:])

    # 1. Extract physical edges and find components
    p1_edges = _get_physical_edges(p1.routes)
    p2_edges = _get_physical_edges(p2.routes)
    uncommon_edges = (p1_edges | p2_edges) - (p1_edges & p2_edges)
    components = _get_components_from_uncommon_edges(p1.giant_tour, p2.giant_tour, uncommon_edges)

    # 2. Inherit components
    child_routes, child_nodes = _inherit_components(components, p1, p2, wastes, capacity, rng)

    # 3. Pack remaining nodes
    all_nodes = set(p1.giant_tour) | set(p2.giant_tour)
    pool = list(all_nodes - child_nodes)
    rng.shuffle(pool)
    loads = [sum(wastes.get(n, 0) for n in r) for r in child_routes]
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()

    _greedy_pack_pool(pool, child_routes, child_nodes, loads, dist_matrix, wastes, capacity, R, C, mandatory_set)

    # 4. Enforce mandatory nodes
    if mandatory_nodes:
        _enforce_mandatory_nodes(child_routes, mandatory_nodes, wastes, capacity, loads)

    # Reconstruct a full giant tour: visited nodes first, then unvisited.
    # Fix 17: Shuffle unvisited suffix to remove Split evaluation bias.
    visited_set = {node for route in child_routes for node in route}
    route_nodes = [node for route in child_routes for node in route]
    unvisited = [n for n in p1.giant_tour if n not in visited_set]
    rng.shuffle(unvisited)
    child_gt = route_nodes + unvisited

    return _get_individual_class()(child_gt, expand_pool=p1.expand_pool)
