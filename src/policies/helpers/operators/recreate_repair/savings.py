"""
Savings-Based Insertion Operator Module.

This module implements a hybrid insertion heuristic for the VRPP. It combines
the Clarke & Wright savings principle with economic profitability checks.
It ensures that mandatory nodes are serves at any cost while opportunistic
nodes are only inserted if their marginal revenue exceeds their detour cost.

Reference:
    Mole, R. H., & Jameson, S. R. (1976). "A sequential algorithm for the
    routing of delivery vehicles". OR, 27(2), 237-248.

Score metric: S = (d(0,u) + d(u,0)) - (d(i,u) + d(u,j) - d(i,j))
Constraint: savings S must be maximized.
"""

from typing import Dict, List, Optional

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def savings_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Insert removed nodes based on maximum savings versus a dedicated route.

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: Dictionary mapping node ID to waste volume.
        capacity: Maximum vehicle capacity.
        mandatory_nodes: Nodes that must be routed regardless of profit.

    Returns:
        List[List[int]]: Updated routes.

    Note:
        Implements Sequential Savings (Mole & Jameson 1976) adapted for VRP:
        1. For each node, calculate the "insertion savings" S = dedicated_cost - detour_cost.
        2. Insert the node at the position with the maximum saving.
        3. Mandatory nodes are prioritized with a large score constant.
        4. If no existing route has capacity, open a new route.
    """
    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = removed_nodes.copy()

    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()

    loads = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    while unassigned:
        best_node = -1
        best_route_idx = -1
        best_pos = -1
        best_saving = float("-inf")

        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            node_waste = wastes.get(node, 0.0)

            # The cost of serving this node alone (depot -> node -> depot)
            dedicated_cost = dist_matrix[0, node] + dist_matrix[node, 0]

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    # Detour cost of inserting 'node' between 'prev' and 'nxt'
                    detour_cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    # Savings: Distance of dedicated route minus the detour distance
                    saving = dedicated_cost - detour_cost

                    # Heavily prioritize mandatory nodes to ensure feasibility
                    effective_saving = saving + (1e9 if is_mandatory else 0)

                    if effective_saving > best_saving:
                        best_saving = effective_saving
                        best_node = node
                        best_route_idx = r_idx
                        best_pos = pos

        if best_node != -1:
            routes[best_route_idx].insert(best_pos, best_node)
            loads[best_route_idx] += wastes.get(best_node, 0.0)
            unassigned.remove(best_node)
        elif unassigned:
            # Fallback: if no existing route has capacity, open a new route
            # for the first remaining node (prioritizing mandatory).
            candidates = sorted(list(unassigned), key=lambda x: 0 if x in mandatory_nodes_set else 1)
            node = candidates[0]
            if wastes.get(node, 0.0) <= capacity or node in mandatory_nodes_set:
                routes.append([node])
                loads.append(wastes.get(node, 0.0))
                unassigned.remove(node)
            else:
                # Exceeds capacity and not mandatory
                unassigned.remove(node)
        else:
            break

    return routes


def savings_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    depot: int = 0,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Insert nodes using the Profit-Aware Clarke-Wright (PACW) principle.

    Score metric: S = (d(0,u) + d(u,0)) - (d(i,u) + d(u,j) - d(i,j))
    Constraint: serving node 'u' must yield (w_u * R) - (detour_cost * C) > 0

    Args:
        routes: List of current routes (sequences of node IDs).
        removed_nodes: Nodes waiting to be re-inserted.
        dist_matrix: 2D array of travel distances.
        wastes: Map of node ID to current waste volume (kg).
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier (currency/kg).
        C: Cost multiplier (currency/km).
        mandatory_nodes: Nodes that MUST be served regardless of profit.
        depot: The index representing the depot (default 0).

    Returns:
        List[List[int]]: Solution with nodes re-inserted into routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = set(range(1, n_nodes + 1)) - visited
    else:
        unassigned = set(removed_nodes)

    # Pre-calculate loads to avoid repeated sum() calls
    loads = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    while unassigned:
        best_score = float("-inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0.0)
            is_mandatory = node in mandatory_nodes_set
            revenue = node_waste * R

            # Distance of serving this node on a dedicated route from depot
            dedicated_dist = dist_matrix[depot, node] + dist_matrix[node, depot]

            for r_idx, route in enumerate(routes):
                # Capacity Constraint
                if loads[r_idx] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else depot
                    nxt = route[pos] if pos < len(route) else depot

                    # Marginal increase in distance
                    detour_dist = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

                    # Profit of this specific insertion
                    insertion_profit = revenue - (detour_dist * C)

                    # Economic Termination: Skip if insertion is a net loss (only for non-mandatory)
                    if not is_mandatory and insertion_profit < -1e-4:
                        continue

                    # Savings logic: How much better is this detour than a dedicated route?
                    # S = Dedicated trip distance - Detour distance
                    saving = dedicated_dist - detour_dist

                    # Weighting: Prioritize mandatory nodes via large constant
                    # Weighting: For opportunistic nodes, use distance savings
                    score = saving + (1e9 if is_mandatory else 0)

                    if score > best_score:
                        best_score = score
                        best_node = node
                        best_route_idx = r_idx
                        best_pos = pos

            # Evaluate new route (Speculative Seeding)
            new_cost = dist_matrix[depot, node] + dist_matrix[node, depot]
            new_profit = revenue - (new_cost * C)
            seed_hurdle = -0.5 * (new_cost * C)

            if is_mandatory or new_profit >= seed_hurdle:
                # PACW savings: S = dedicated_dist - detour_dist
                new_score = 0 + (1e9 if is_mandatory else 0)
                if new_score > best_score:
                    best_score = new_score
                    best_node = node
                    best_route_idx = len(routes)
                    best_pos = 0

        # Execute insertion if a valid candidate was found
        if best_node != -1:
            if best_route_idx == len(routes):
                routes.append([best_node])
                loads.append(wastes.get(best_node, 0.0))
            else:
                routes[best_route_idx].insert(best_pos, best_node)
                loads[best_route_idx] += wastes.get(best_node, 0.0)
            unassigned.remove(best_node)
        else:
            # Fallback: Check if we have mandatory nodes that couldn't fit in existing routes
            mandatory_remaining = sorted([n for n in unassigned if n in mandatory_nodes_set])
            if mandatory_remaining:
                # Open a new route for the first remaining mandatory node
                new_node = mandatory_remaining[0]
                routes.append([new_node])
                loads.append(wastes.get(new_node, 0.0))
                unassigned.remove(new_node)
            else:
                # No more profitable opportunistic moves or mandatory nodes exist.
                break

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
