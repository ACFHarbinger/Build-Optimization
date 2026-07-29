"""
Greedy Insertion Operator Module.

This module implements the greedy insertion heuristic, which iteratively inserts
unassigned nodes into the position that minimizes the immediate cost increase.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.greedy import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def greedy_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes into their best (cheapest) positions greedily.

    Efficiency: O(U * N * L_avg) incremental evaluation using a RoutingCache.
    Only re-evaluates the modified route after each insertion instead of recalculating
    all possible options (O(U^2 * N * L_avg)).

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.
        noise: Noise level for cost perturbation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    # Calculate current loads and track visited nodes
    loads = []
    visited = set()
    for route in routes:
        loads.append(sum(wastes.get(node, 0) for node in route))
        visited.update(route)

    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned_set = set(range(1, n_nodes + 1)) - visited
    else:
        unassigned_set = set(removed_nodes)
        if mandatory_nodes:
            unassigned_set.update(set(mandatory_nodes) - visited)

    unassigned = sorted(list(unassigned_set))
    if not unassigned:
        return routes

    # --- Routing Cache Initialization ---
    # node_route_cache[node][r_idx] = (cost, pos)
    node_route_cache: Dict[int, List[Tuple[float, int]]] = {}
    # node_overall_best[node] = (cost, r_idx, pos)
    node_overall_best: Dict[int, Tuple[float, int, int]] = {}

    def get_best_for_route(node_id: int, r_idx: int) -> Tuple[float, int]:
        """Find best insertion position for a node in a specific route."""
        route = routes[r_idx]
        node_waste = wastes.get(node_id, 0)
        if loads[r_idx] + node_waste > capacity:
            return float("inf"), -1

        best_r_cost = float("inf")
        best_r_pos = -1

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost = dist_matrix[prev, node_id] + dist_matrix[node_id, nxt] - dist_matrix[prev, nxt]
            if noise != 0:
                cost = max(0.0, cost + noise)

            if cost < best_r_cost:
                best_r_cost = cost
                best_r_pos = pos
        return best_r_cost, best_r_pos

    # Initial population
    for node in unassigned:
        node_route_cache[node] = []
        best_node_cost = float("inf")
        best_node_r_idx = -1
        best_node_pos = -1

        for i in range(len(routes)):
            cost, pos = get_best_for_route(node, i)
            node_route_cache[node].append((cost, pos))
            if cost < best_node_cost:
                best_node_cost = cost
                best_node_r_idx = i
                best_node_pos = pos
        node_overall_best[node] = (best_node_cost, best_node_r_idx, best_node_pos)

    # --- Main Loop ---
    while unassigned:
        # Find globally best insertion
        best_cost = float("inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            cost, r_idx, pos = node_overall_best[node]
            if cost < best_cost:
                best_cost = cost
                best_node = node
                best_route_idx = r_idx
                best_pos = pos

        if best_node != -1:
            # Apply insertion
            routes[best_route_idx].insert(best_pos, best_node)
            loads[best_route_idx] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
            del node_route_cache[best_node]
            del node_overall_best[best_node]

            # Incremental Update: Re-evaluate modified route for all remaining nodes
            for node in unassigned:
                # Update modified route cache
                new_r_cost, new_r_pos = get_best_for_route(node, best_route_idx)
                old_r_cost, _ = node_route_cache[node][best_route_idx]
                node_route_cache[node][best_route_idx] = (new_r_cost, new_r_pos)

                # Check if overall best needs update
                curr_best_cost, curr_r_idx, _ = node_overall_best[node]

                if curr_r_idx == best_route_idx or new_r_cost < curr_best_cost:
                    # Full re-scan of routes for this node (since its best route was modified)
                    # or it found a new better cost in the modified route.
                    best_n_cost = float("inf")
                    best_n_r_idx = -1
                    best_n_pos = -1
                    for i, (c, p) in enumerate(node_route_cache[node]):
                        if c < best_n_cost:
                            best_n_cost = c
                            best_n_r_idx = i
                            best_n_pos = p
                    node_overall_best[node] = (best_n_cost, best_n_r_idx, best_n_pos)
        else:
            # Handle fallback (force new route)
            node = unassigned[0]
            routes.append([node])
            loads.append(wastes.get(node, 0))
            unassigned.remove(node)
            # When a new route is added, all nodes technically have a new option.
            # But in greedy_insertion (non-profit), starting a new route is a fallback
            # and usually only happens when no existing route is feasible.
            # However, for completeness, we should update the cache.
            new_r_idx = len(routes) - 1
            for u_node in unassigned:
                if u_node in node_route_cache:  # if not removed
                    cost, pos = get_best_for_route(u_node, new_r_idx)
                    node_route_cache[u_node].append((cost, pos))
                    # Update overall best if this new route is better (unlikely in greedy cost)
                    if cost < node_overall_best[u_node][0]:
                        node_overall_best[u_node] = (cost, new_r_idx, pos)

    return routes


def greedy_profit_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Greedily insert nodes to maximize profit (revenue - cost).

    Efficiency: O(U * N * L_avg) incremental evaluation using a RoutingCache.
    Instead of minimizing cost, we maximize (waste * R - delta_dist * C).
    Nodes with negative max profit are skipped unless they are mandatory.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier (per waste unit).
        C: Cost multiplier (per distance unit).
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, all unvisited nodes are considered.
        noise: Noise level for cost perturbation.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0) for node in r) for r in routes]

    visited = set()
    for r in routes:
        visited.update(r)

    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unassigned_set = set(range(1, n_nodes + 1)) - visited
    else:
        unassigned_set = set(removed_nodes)
        if mandatory_nodes:
            unassigned_set.update(set(mandatory_nodes) - visited)

    unassigned = sorted(list(unassigned_set))
    if not unassigned:
        return routes

    # --- Routing Cache Initialization ---
    # node_route_cache[node][r_idx] = (profit, pos)
    node_route_cache: Dict[int, List[Tuple[float, int]]] = {}
    # node_overall_best[node] = (profit, r_idx, pos)
    node_overall_best: Dict[int, Tuple[float, int, int]] = {}

    def get_best_for_route(node_id: int, r_idx: int) -> Tuple[float, int]:
        """Find best insertion position (profit) for a node in an existing route."""
        route = routes[r_idx]
        node_waste = wastes.get(node_id, 0)
        if loads[r_idx] + node_waste > capacity:
            return -float("inf"), -1

        revenue = node_waste * R
        is_mandatory = node_id in mandatory_nodes_set
        best_r_profit = -float("inf")
        best_r_pos = -1

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost = dist_matrix[prev, node_id] + dist_matrix[node_id, nxt] - dist_matrix[prev, nxt]
            if noise != 0:
                cost = max(0.0, cost + noise)
            profit = revenue - (cost * C)

            # Skip unprofitable positions for non-mandatory nodes before comparison
            if not is_mandatory and profit < -1e-4:
                continue

            effective_profit = profit + (1e9 if is_mandatory else 0)
            if effective_profit > best_r_profit:
                best_r_profit = effective_profit
                best_r_pos = pos

        return best_r_profit, best_r_pos

    def get_seed_profit(node_id: int) -> float:
        """Calculate speculative seeding profit for a new route."""
        node_waste = wastes.get(node_id, 0)
        revenue = node_waste * R
        new_cost = dist_matrix[0, node_id] + dist_matrix[node_id, 0]
        if noise != 0:
            new_cost = max(0.0, new_cost + noise)
        new_profit = revenue - (new_cost * C)
        seed_hurdle = -0.5 * (new_cost * C)
        is_mandatory = node_id in mandatory_nodes_set

        if is_mandatory or new_profit >= seed_hurdle:
            return new_profit + (1e9 if is_mandatory else 0)
        return -float("inf")

    # Initial population
    for node in unassigned:
        node_route_cache[node] = []
        best_n_profit = -float("inf")
        best_n_r_idx = -1
        best_n_pos = -1

        # Check existing routes
        for i in range(len(routes)):
            profit, pos = get_best_for_route(node, i)
            node_route_cache[node].append((profit, pos))
            if profit > best_n_profit:
                best_n_profit = profit
                best_n_r_idx = i
                best_n_pos = pos

        # Check new route (seeding)
        seed_profit = get_seed_profit(node)
        if seed_profit > best_n_profit:
            best_n_profit = seed_profit
            best_n_r_idx = len(routes)
            best_n_pos = 0

        node_overall_best[node] = (best_n_profit, best_n_r_idx, best_n_pos)

    # --- Main Loop ---
    while unassigned:
        best_profit = -float("inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            profit, r_idx, pos = node_overall_best[node]
            if profit > best_profit:
                best_profit = profit
                best_node = node
                best_route_idx = r_idx
                best_pos = pos

        if best_node != -1:
            if best_route_idx == len(routes):
                # Speculative Seeding: Open New Route
                routes.append([best_node])
                loads.append(wastes.get(best_node, 0))
                unassigned.remove(best_node)
                del node_route_cache[best_node]
                del node_overall_best[best_node]

                # Update ALL remaining nodes for the new route
                new_r_idx = len(routes) - 1
                for u_node in unassigned:
                    # Update cache with the previously 'new' route (now index new_r_idx)
                    # We need to re-evaluate it as a standard route now
                    profit, pos = get_best_for_route(u_node, new_r_idx)
                    node_route_cache[u_node].append((profit, pos))

                    # Re-calculate seed option for the 'next' new route (empty again)
                    new_seed_profit = get_seed_profit(u_node)

                    # Scan cache + new seed to find overall best
                    curr_max_profit = new_seed_profit
                    curr_max_r_idx = len(routes)
                    curr_max_pos = 0

                    for i, (p, pos_i) in enumerate(node_route_cache[u_node]):
                        if p > curr_max_profit:
                            curr_max_profit = p
                            curr_max_r_idx = i
                            curr_max_pos = pos_i
                    node_overall_best[u_node] = (curr_max_profit, curr_max_r_idx, curr_max_pos)
            else:
                # Standard Insertion
                routes[best_route_idx].insert(best_pos, best_node)
                loads[best_route_idx] += wastes.get(best_node, 0)
                unassigned.remove(best_node)
                del node_route_cache[best_node]
                del node_overall_best[best_node]

                # Incremental Update: Re-evaluate modified route for all remaining nodes
                for u_node in unassigned:
                    new_p, new_p_pos = get_best_for_route(u_node, best_route_idx)
                    node_route_cache[u_node][best_route_idx] = (new_p, new_p_pos)

                    curr_best_p, curr_r_idx, _ = node_overall_best[u_node]
                    if curr_r_idx == best_route_idx or new_p > curr_best_p:
                        # Full scan required
                        best_of_n = get_seed_profit(u_node)
                        best_of_n_r_idx = len(routes)
                        best_of_n_ppos = 0
                        for i, (p, pos_i) in enumerate(node_route_cache[u_node]):
                            if p > best_of_n:
                                best_of_n = p
                                best_of_n_r_idx = i
                                best_of_n_ppos = pos_i
                        node_overall_best[u_node] = (best_of_n, best_of_n_r_idx, best_of_n_ppos)
        else:
            # Handle remaining mandatory nodes
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                # Update cache for new route (similar to seed logic above)
                new_r_idx = len(routes) - 1
                for u_node in unassigned:
                    if u_node in node_route_cache:
                        p, ppos = get_best_for_route(u_node, new_r_idx)
                        node_route_cache[u_node].append((p, ppos))
                        seed_p = get_seed_profit(u_node)
                        # Re-scan
                        max_p = seed_p
                        max_r = len(routes)
                        max_pos = 0
                        for i, (pi, pposi) in enumerate(node_route_cache[u_node]):
                            if pi > max_p:
                                max_p = pi
                                max_r = i
                                max_pos = pposi
                        node_overall_best[u_node] = (max_p, max_r, max_pos)
            else:
                break

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
