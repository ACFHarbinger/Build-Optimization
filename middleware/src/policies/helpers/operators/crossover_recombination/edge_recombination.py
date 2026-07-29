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
    """Lazy import of Individual to avoid circular import at module init."""
    from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual import Individual

    return Individual


def edge_recombination_crossover(
    p1: Individual,
    p2: Individual,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Edge Recombination Crossover (ERX): Preserves edge adjacencies.

    Algorithm:
        1. Build edge adjacency table from decoded physical routes of both parents.
           Unvisited nodes (those not in any route) have empty adjacency sets and
           are placed via the random fallback, preserving the full genotype.
        2. Start from a uniformly random node.
        3. Iteratively select next node with fewest remaining edges.
        4. Update adjacency table after each selection.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.
        mandatory_nodes: List of nodes that MUST be visited.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    # All nodes in the genotype (visited + unvisited).
    all_nodes = list(set(p1.giant_tour) | set(p2.giant_tour))
    if not all_nodes:
        return _get_individual_class()([])

    adj_table: Dict[int, Set[int]] = defaultdict(set)

    def add_route_edges(routes: List[List[int]]) -> None:
        """Add undirected edges from decoded physical routes."""
        for route in routes:
            if not route:
                continue
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                adj_table[u].add(v)
                adj_table[v].add(u)

    # Use decoded routes when available, otherwise skip edge building.
    # Unvisited nodes will have empty adjacency and trigger the random fallback.
    if p1.routes:
        add_route_edges(p1.routes)
    if p2.routes:
        add_route_edges(p2.routes)

    # Fix 12: Start from a random node instead of biased candidates.
    current = rng.choice(all_nodes)

    child_gt: List[int] = []
    remaining = set(all_nodes)
    remaining.discard(current)
    child_gt.append(current)

    # Build tour by selecting nodes with fewest edges
    while remaining:
        # Get neighbors of current node
        neighbors = adj_table[current] & remaining

        if neighbors:
            # Select neighbor with fewest edges (ties broken randomly)
            candidates = [(len(adj_table[n] & remaining), rng.random(), n) for n in neighbors]
            next_node = min(candidates)[2]
        else:
            # No neighbors available, select random remaining node
            next_node = rng.choice(list(remaining))

        child_gt.append(next_node)
        remaining.discard(next_node)

        # Remove selected node from all adjacency lists
        for node in remaining:
            adj_table[node].discard(next_node)

        current = next_node

    return _get_individual_class()(child_gt, expand_pool=p1.expand_pool)


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


def capacity_aware_erx(
    p1: Individual,
    p2: Individual,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
    mandatory_nodes: Optional[List[int]] = None,
) -> Individual:
    """
    Capacity-Aware Edge Recombination Crossover (C-ERX).
    Adapts the classical ERX for VRPP by walking the physical adjacency matrix of
    both parents while strictly enforcing vehicle capacity and using profit tie-breakers.
    """
    if rng is None:
        rng = random.Random()

    if not p1.routes or not p2.routes:
        return _get_individual_class()(p1.giant_tour[:])

    # 1. Build Union Adjacency from physical routes
    all_edges = _get_physical_edges(p1.routes) | _get_physical_edges(p2.routes)
    adj_table: Dict[int, Set[int]] = defaultdict(set)

    for u, v in all_edges:
        if u != 0 and v != 0:
            adj_table[u].add(v)
            adj_table[v].add(u)

    all_nodes = list(set(p1.giant_tour) | set(p2.giant_tour))
    if not all_nodes:
        return _get_individual_class()([])

    child_routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0.0
    remaining = set(all_nodes)
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()

    # The Capacity Walk
    while remaining:
        # Start a new route if the current one is empty
        if not current_route:
            candidate = rng.choice(list(remaining))
            solo_cost = (dist_matrix[0, candidate] + dist_matrix[candidate, 0]) * C
            solo_revenue = wastes.get(candidate, 0.0) * R
            if solo_revenue < solo_cost and candidate not in mandatory_nodes_set:
                remaining.discard(candidate)
                continue
            current_route.append(candidate)
            current_load += wastes.get(candidate, 0.0)
            remaining.discard(candidate)

        current_node = current_route[-1]  # always set from current route state
        raw_neighbors = adj_table[current_node] & remaining
        valid_neighbors = [n for n in raw_neighbors if current_load + wastes.get(n, 0.0) <= capacity]

        if valid_neighbors:
            candidates = [
                (
                    len(adj_table[n] & remaining),
                    -(wastes.get(n, 0.0) * R - dist_matrix[current_node, n] * C),
                    rng.random(),
                    n,
                )
                for n in valid_neighbors
            ]
            next_node = min(candidates)[3]
            current_route.append(next_node)
            current_load += wastes.get(next_node, 0.0)
            remaining.discard(next_node)
        else:
            child_routes.append(current_route)
            current_route = []
            current_load = 0.0

    if current_route:
        child_routes.append(current_route)

    # Fix 22: Enforce mandatory nodes in visited routes.
    if mandatory_nodes:
        visited_in_routes = {n for route in child_routes for n in route}
        missing_mandatory = mandatory_nodes_set - visited_in_routes
        loads = [sum(wastes.get(n, 0.0) for n in r) for r in child_routes]

        for node in missing_mandatory:
            node_waste = wastes.get(node, 0.0)
            best_r = min(range(len(child_routes)), key=lambda i: loads[i], default=None)
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
            f"Mandatory nodes missing from routes: {mandatory_nodes_set - visited_prefix}"
        )

    # Reconstruct a full-length giant tour: visited nodes first, then unvisited.
    # Fix 17: Shuffle unvisited suffix to remove Split evaluation bias.
    visited_set = {node for route in child_routes for node in route}
    route_nodes = [node for route in child_routes for node in route]
    unvisited = [n for n in p1.giant_tour if n not in visited_set]
    rng.shuffle(unvisited)
    child_gt = route_nodes + unvisited

    return _get_individual_class()(child_gt, expand_pool=p1.expand_pool)
