"""
Nearest Insertion Repair Operator Module.

This module implements the nearest insertion heuristic, a classical TSP construction
method adapted for VRP. It iteratively selects the unassigned node nearest to
any node currently in the tour and inserts it at the position that minimizes the
cost increase.

Reference:
    Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M. (1977).
    "An analysis of several heuristics for the traveling salesman problem".
    SIAM Journal on Computing, 6(3), 563-581.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.nearest import nearest_insertion
    >>> routes = nearest_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def _get_nearest_node(
    unassigned: List[int],
    routes: List[List[int]],
    dist_matrix: np.ndarray,
) -> Optional[int]:
    """Find the unassigned node nearest to any current route or depot."""
    nearest_node = None
    min_distance = float("inf")

    for node in unassigned:
        # Distance to existing nodes in routes
        for route in routes:
            for route_node in route:
                d = dist_matrix[node, route_node]
                if d < min_distance:
                    min_distance = d
                    nearest_node = node
        # Distance to depot
        d_depot = dist_matrix[node, 0]
        if d_depot < min_distance:
            min_distance = d_depot
            nearest_node = node

    return nearest_node


def _find_cheapest_insertion(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    capacity: float,
    node_waste: float,
    revenue: float,
    is_mandatory: bool,
    R: Optional[float] = None,
    C: float = 1.0,
) -> Tuple[int, int, float]:
    """Find the cheapest insertion position for a node."""
    best_cost = float("inf")
    best_route_idx = -1
    best_pos = -1

    for i, route in enumerate(routes):
        if loads[i] + node_waste > capacity:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0

            cost_increase = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

            # Profitability check for VRPP
            if R is not None and not is_mandatory and cost_increase * C > revenue:
                continue

            if cost_increase < best_cost:
                best_cost = cost_increase
                best_route_idx = i
                best_pos = pos

    return best_route_idx, best_pos, best_cost


def nearest_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
) -> List[List[int]]:
    """
    Insert removed nodes using the nearest insertion heuristic.

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Symmetric distance matrix.
        wastes: Dictionary mapping node index to waste/demand.
        capacity: Maximum vehicle capacity.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.

    Returns:
        List[List[int]]: Updated routes.

    Note:
        Implements Nearest Insertion (Rosenkrantz et al. 1977) adapted for VRP:
        1. Find node *u* in *unassigned* nearest to any node already in a route.
        2. Find cheapest insertion position for *u* across all current routes.
        3. Insert *u* in the best position found, or open a new route if no
           feasible insertion exists or the new route is cheaper.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0.0) for node in route) for route in routes]

    visited = {node for route in routes for node in route}
    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        nearest_node = _get_nearest_node(unassigned, routes, dist_matrix)
        if nearest_node is None:
            break

        node_waste = wastes.get(nearest_node, 0.0)

        best_route_idx, best_pos, best_cost = _find_cheapest_insertion_dist(
            nearest_node, routes, loads, dist_matrix, capacity, node_waste
        )

        new_route_cost = dist_matrix[0, nearest_node] + dist_matrix[nearest_node, 0]

        # Decision logic: Compare Existing vs New (Rosenkrantz et al. 1977 adaptation for VRP)
        if best_route_idx != -1 and best_cost <= new_route_cost:
            routes[best_route_idx].insert(best_pos, nearest_node)
            loads[best_route_idx] += node_waste
            unassigned.remove(nearest_node)
        elif node_waste <= capacity:
            # Open new route if no existing route has capacity or dedicated is cheaper
            routes.append([nearest_node])
            loads.append(node_waste)
            unassigned.remove(nearest_node)
        elif nearest_node in mandatory_set:
            # Fallback for mandatory nodes even if they exceed capacity (last resort)
            routes.append([nearest_node])
            loads.append(node_waste)
            unassigned.remove(nearest_node)
        else:
            # Node is uninsertable (exceeds capacity) and not mandatory
            unassigned.remove(nearest_node)

    return routes


def _find_cheapest_insertion_dist(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    capacity: float,
    node_waste: float,
) -> Tuple[int, int, float]:
    """Find the cheapest insertion position for a node (distance only)."""
    best_cost = float("inf")
    best_route_idx = -1
    best_pos = -1

    for i, route in enumerate(routes):
        if loads[i] + node_waste > capacity:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0

            cost_increase = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

            if cost_increase < best_cost:
                best_cost = cost_increase
                best_route_idx = i
                best_pos = pos

    return best_route_idx, best_pos, best_cost


def nearest_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Nearest insertion with explicit profit maximization for VRPP.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0.0) for node in route) for route in routes]

    visited = {node for route in routes for node in route}
    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        node_distances = []
        for node in unassigned:
            min_d = float("inf")
            for route in routes:
                for r_node in route:
                    min_d = min(min_d, dist_matrix[node, r_node])
            min_d = min(min_d, dist_matrix[node, 0])
            node_distances.append((node, min_d))

        node_distances.sort(key=lambda x: x[1])

        inserted_any = False
        for node, _ in node_distances:
            node_waste = wastes.get(node, 0.0)
            revenue = node_waste * R
            is_mandatory = node in mandatory_set

            best_route_idx, best_pos, best_cost = _find_cheapest_insertion(
                node, routes, loads, dist_matrix, capacity, node_waste, revenue, is_mandatory, R, C
            )

            new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
            new_profit = revenue - (new_cost * C)
            seed_hurdle = -0.5 * (new_cost * C)

            # Decision logic: Comparison of Existing vs New vs Skip
            if best_route_idx != -1:
                # Calculate cost_increase to compute profit for existing route
                prev = routes[best_route_idx][best_pos - 1] if best_pos > 0 else 0
                nxt = routes[best_route_idx][best_pos] if best_pos < len(routes[best_route_idx]) else 0
                cost_increase = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                existing_profit = revenue - (cost_increase * C)

                # If new route is better (and passes seed hurdle)
                if new_profit > existing_profit and (is_mandatory or new_profit >= seed_hurdle):
                    routes.append([node])
                    loads.append(node_waste)
                    unassigned.remove(node)
                    inserted_any = True
                    break
                else:
                    # Use existing route
                    routes[best_route_idx].insert(best_pos, node)
                    loads[best_route_idx] += node_waste
                    unassigned.remove(node)
                    inserted_any = True
                    break
            elif is_mandatory or new_profit >= seed_hurdle:
                # Only NEW route is an option
                routes.append([node])
                loads.append(node_waste)
                unassigned.remove(node)
                inserted_any = True
                break

        if not inserted_any:
            break

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
