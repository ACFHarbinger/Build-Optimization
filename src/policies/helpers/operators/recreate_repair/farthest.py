"""
Farthest Insertion Repair Operator Module.

This module implements the farthest insertion heuristic, a classical TSP construction
method adapted for VRP. It iteratively selects the unassigned node farthest from the
current tour and inserts it at the position that minimizes the cost increase.

This heuristic is particularly effective for creating geographically diverse tours
and avoiding local clustering, which can be beneficial for VRPP instances.

Reference:
    Rosenkrantz, D. J., Stearns, R. E., & Lewis, P. M. (1977).
    "An analysis of several heuristics for the traveling salesman problem".
    SIAM Journal on Computing, 6(3), 563-581.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.farthest import farthest_insertion
    >>> routes = farthest_insertion(routes, removed, dist_matrix, wastes, capacity, R, C)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def _get_farthest_node(
    unassigned: List[int],
    routes: List[List[int]],
    dist_matrix: np.ndarray,
) -> Optional[int]:
    """Find the unassigned node farthest from any current route or depot."""
    farthest_node = None
    max_min_distance = -1.0

    for node in unassigned:
        min_distance = float("inf")
        # Distance to existing nodes in routes
        for route in routes:
            for route_node in route:
                min_distance = min(min_distance, dist_matrix[node, route_node])
        # Distance to depot
        min_distance = min(min_distance, dist_matrix[node, 0])

        if min_distance > max_min_distance:
            max_min_distance = min_distance
            farthest_node = node
    return farthest_node


def _find_cheapest_insertion(
    farthest_node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    capacity: float,
    node_waste: float,
    revenue: float,
    is_mandatory: bool,
    R: Optional[float] = None,
    C: Optional[float] = None,
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

            cost_increase = dist_matrix[prev, farthest_node] + dist_matrix[farthest_node, nxt] - dist_matrix[prev, nxt]

            if R is not None and C is not None and not is_mandatory and cost_increase * C > revenue:
                continue

            if cost_increase < best_cost:
                best_cost = cost_increase
                best_route_idx = i
                best_pos = pos

    return best_route_idx, best_pos, best_cost


def _find_nearest_insertion(
    farthest_node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    capacity: float,
    node_waste: float,
) -> Tuple[int, int, float]:
    """Find the nearest insertion position for a node."""
    best_cost = float("inf")
    best_route_idx = -1
    best_pos = -1

    for i, route in enumerate(routes):
        if loads[i] + node_waste > capacity:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0

            cost_increase = dist_matrix[prev, farthest_node] + dist_matrix[farthest_node, nxt] - dist_matrix[prev, nxt]

            if cost_increase < best_cost:
                best_cost = cost_increase
                best_route_idx = i
                best_pos = pos

    return best_route_idx, best_pos, best_cost


def farthest_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
) -> List[List[int]]:
    """
    Insert removed nodes using the farthest insertion heuristic.

    The algorithm works as follows:
    1. For each unassigned node, compute its minimum distance to any node in any route
    2. Select the node with the maximum such distance (farthest node)
    3. Insert it at the position that minimizes cost increase (cheapest insertion)
    4. Repeat until all nodes are inserted or profitability constraints prevent further insertion

    This creates geographically diverse tours and helps avoid premature clustering.

    Args:
        routes: Partial routes (list of node sequences).
        removed_nodes: List of unassigned node indices.
        dist_matrix: Symmetric distance matrix (N+1 x N+1, including depot at index 0).
        wastes: Dictionary mapping node index to waste/demand.
        capacity: Maximum vehicle capacity.
        mandatory_nodes: List of mandatory node indices that must be visited.
        expand_pool: If True, consider all unvisited nodes; if False, only removed_nodes.

    Returns:
        List[List[int]]: Updated routes after farthest insertion.

    Note:
        Implements Farthest Insertion (Rosenkrantz et al. 1977) adapted for VRP:
        1. Find node *u* in *unassigned* farthest from any node already in a route.
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
        farthest_node = _get_farthest_node(unassigned, routes, dist_matrix)
        if farthest_node is None:
            break

        node_waste = wastes.get(farthest_node, 0.0)
        is_mandatory = farthest_node in mandatory_set

        best_route_idx, best_pos, best_cost = _find_nearest_insertion(
            farthest_node, routes, loads, dist_matrix, capacity, node_waste
        )

        new_route_cost = dist_matrix[0, farthest_node] + dist_matrix[farthest_node, 0]

        # Decision logic: Compare Existing vs New (Rosenkrantz et al. 1977 adaptation for VRP)
        if best_route_idx != -1 and best_cost <= new_route_cost:
            routes[best_route_idx].insert(best_pos, farthest_node)
            loads[best_route_idx] += node_waste
            unassigned.remove(farthest_node)
        elif node_waste <= capacity:
            # Open new route if no existing route has capacity or dedicated is cheaper
            routes.append([farthest_node])
            loads.append(node_waste)
            unassigned.remove(farthest_node)
        elif is_mandatory:
            # Fallback for mandatory nodes
            routes.append([farthest_node])
            loads.append(node_waste)
            unassigned.remove(farthest_node)
        else:
            # Node is uninsertable and not mandatory
            unassigned.remove(farthest_node)

    return routes


def farthest_profit_insertion(
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
    Farthest insertion with explicit profit maximization for VRPP.

    Similar to farthest_insertion, but selects the farthest node that can
    still be inserted profitably (revenue > cost increase).

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: Waste/demand dictionary.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory_nodes: Mandatory node indices.
        expand_pool: Consider all unvisited nodes if True.

    Returns:
        List[List[int]]: Updated routes.
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
        # Compute distances for all unassigned nodes
        node_distances = []
        for node in unassigned:
            min_distance = float("inf")
            for route in routes:
                for route_node in route:
                    min_distance = min(min_distance, dist_matrix[node, route_node])
            min_distance = min(min_distance, dist_matrix[node, 0])
            node_distances.append((node, min_distance))

        # Sort by distance descending (farthest first)
        node_distances.sort(key=lambda x: x[1], reverse=True)

        inserted_any = False
        for farthest_node, _ in node_distances:
            node_waste = wastes.get(farthest_node, 0.0)
            revenue = node_waste * R
            is_mandatory = farthest_node in mandatory_set

            best_route_idx, best_pos, _ = _find_cheapest_insertion(
                farthest_node, routes, loads, dist_matrix, capacity, node_waste, revenue, is_mandatory, R, C
            )

            new_cost = dist_matrix[0, farthest_node] + dist_matrix[farthest_node, 0]
            new_profit = revenue - (new_cost * C)
            seed_hurdle = -0.5 * (new_cost * C)

            # Compare insertion into existing routes vs. starting a new route
            if best_route_idx != -1:
                # Calculate cost_increase to compute profit for existing route
                prev = routes[best_route_idx][best_pos - 1] if best_pos > 0 else 0
                nxt = routes[best_route_idx][best_pos] if best_pos < len(routes[best_route_idx]) else 0
                cost_increase = (
                    dist_matrix[prev, farthest_node] + dist_matrix[farthest_node, nxt] - dist_matrix[prev, nxt]
                )
                existing_profit = revenue - (cost_increase * C)

                # If new route is better (and passes seed hurdle)
                if new_profit > existing_profit and (is_mandatory or new_profit >= seed_hurdle):
                    routes.append([farthest_node])
                    loads.append(node_waste)
                    unassigned.remove(farthest_node)
                    inserted_any = True
                    break
                else:
                    # Use existing route
                    routes[best_route_idx].insert(best_pos, farthest_node)
                    loads[best_route_idx] += node_waste
                    unassigned.remove(farthest_node)
                    inserted_any = True
                    break
            elif is_mandatory or new_profit >= seed_hurdle:
                routes.append([farthest_node])
                loads.append(node_waste)
                unassigned.remove(farthest_node)
                inserted_any = True
                break

        if not inserted_any:
            break

    return routes
