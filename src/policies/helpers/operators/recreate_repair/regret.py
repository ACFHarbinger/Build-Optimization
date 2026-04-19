"""
Regret Insertion Operator Module.

This module implements the Regret-k insertion heuristic for VRP repair.
It calculates a regret value for each unassigned node based on the cost difference
between its best and k-th best insertion positions.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.regret import RegretInsertion
    >>> operator = RegretInsertion(k=2)
    >>> new_routes = operator.repair(destroyed_routes, unassigned_nodes)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def regret_2_insertion(
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
    Insert removed nodes based on the regret-2 criterion.
    Regret-2 Insertion Heuristic.

    Prioritizes inserting nodes that would be much more expensive to insert
    later (high regret = cost difference between best and 2nd best option).

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: Optional list of mandatory node indices.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=2,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_k_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes using the regret-k heuristic.

    Implements Algorithm 1 of Pisinger & Ropke (2007), *A general heuristic for
    vehicle routing problems*, §3.2.2 (Regret-k insertion):

    1. Compute for each unassigned request *i* its *k* cheapest insertion costs
       across all current routes: ``c_i^1 \u2264 c_i^2 \u2264 ... \u2264 c_i^k``.
    2. Compute regret: ``regret(i) = \u03a3_{h=2}^{k} (c_i^h - c_i^1)``.
       If fewer than *k* routes can accept *i*, the missing differences are set
       to ``max_observed_cost + large_constant`` so that hard-to-place requests
       are strongly prioritised.
    3. Insert the request with the **maximum** regret.
    4. Tie-breaking (paper, §3.2.2): if two nodes share the same regret, insert
       the one with the **minimum** best insertion cost ``c_i^1``.

    Efficiency: O(U * N * L_avg) incremental evaluation using a RoutingCache.
    Only re-evaluates the modified route after each insertion instead of recalculating
    all possible options (O(U^2 * N * L_avg)).

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        k: Regret degree (2, 3, etc.).
        mandatory_nodes: Optional list of mandatory node indices.
        expand_pool: If True, all unvisited nodes are candidates.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    # Calculate current loads and track visited
    loads = []
    visited = set()
    for route in routes:
        loads.append(sum(wastes.get(node, 0) for node in route))
        visited.update(route)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
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

    def get_best_for_route(node_id: int, r_idx: int) -> Tuple[float, int]:
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
        for i in range(len(routes)):
            cost, pos = get_best_for_route(node, i)
            node_route_cache[node].append((cost, pos))

    # --- Main Loop ---
    while unassigned:
        all_candidates = []
        unprofitable_nodes = []

        for node in unassigned:
            # Filter and sort options from cache
            options = sorted([c for c in node_route_cache[node] if c[0] != float("inf")], key=lambda x: x[0])

            if not options:
                is_mandatory = node in mandatory_nodes_set
                if not is_mandatory:
                    unprofitable_nodes.append(node)
                continue

            # Regret: paper §3.2.2 sum formula (sum of differences from best)
            # regret(i) = sum_{j=1}^k (cost(i, x_j) - cost(i, x_1))
            if len(options) > 1:
                # Limit to k best options per paper
                # If k is larger than available routes, use all available
                target_k = min(k, len(options))
                regret = sum(options[j][0] - options[0][0] for j in range(1, target_k))
                if len(options) < k:
                    # Ropke & Pisinger: if node can't be inserted into k routes,
                    # treat remaining 'missing' routes as having inf cost.
                    # This makes nodes with few insertion options highly prioritized.
                    regret += (k - len(options)) * (max(o[0] for o in options) + 1000.0)
            else:
                regret = 1e9  # Max priority if only 1 (or 0) route possible

            # Find best existing route index for this node
            best_opt = options[0]
            r_idx = -1
            for i, (c, _) in enumerate(node_route_cache[node]):
                if c == best_opt[0]:  # handle ties or just find first match
                    r_idx = i
                    break

            all_candidates.append((regret, node, (best_opt[0], r_idx, best_opt[1])))

        for node in unprofitable_nodes:
            unassigned.remove(node)
            if node in node_route_cache:
                del node_route_cache[node]

        if not all_candidates:
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                # Expand cache for ALL remaining nodes
                new_r_idx = len(routes) - 1
                for u_node in unassigned:
                    cost, pos = get_best_for_route(u_node, new_r_idx)
                    node_route_cache[u_node].append((cost, pos))
                continue
            else:
                break

        # Pick node with max regret.
        # Tie-breaking (Pisinger & Ropke 2007, §3.2.2): if two nodes have equal
        # regret, choose the one with the MINIMUM best insertion cost c_i^1.
        # Sort key: primary = regret descending, secondary = best_cost ascending.
        all_candidates.sort(key=lambda x: (-x[0], x[2][0]))
        _, best_node, (best_cost, r_idx, pos) = all_candidates[0]

        # Apply insertion
        routes[r_idx].insert(pos, best_node)
        loads[r_idx] += wastes.get(best_node, 0)
        unassigned.remove(best_node)
        del node_route_cache[best_node]

        # Incremental Update
        for u_node in unassigned:
            new_p_cost, new_p_pos = get_best_for_route(u_node, r_idx)
            node_route_cache[u_node][r_idx] = (new_p_cost, new_p_pos)

    return routes


def _get_insertion_options_with_profit(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    is_mandatory: bool,
    noise: float,
    max_dist: float,
) -> List[Tuple[float, int, int]]:
    """Helper to calculate insertion options for a node with profit logic."""
    node_waste = wastes.get(node, 0)
    revenue = node_waste * R
    node_options = []

    for i, route in enumerate(routes):
        if loads[i] + node_waste > capacity:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0

            cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

            if noise != 0:
                cost = max(0.0, cost + noise)

            profit = revenue - cost * C

            # Requirement: profit > 0 or is_mandatory
            if is_mandatory or profit > -1e-4:
                node_options.append((profit, i, pos))

    # Evaluate new route (Speculative Seeding Heuristic)
    # See greedy_profit_insertion for theoretical justification
    new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
    if noise != 0:
        new_cost = max(0.0, new_cost + noise)
    new_profit = revenue - (new_cost * C)
    seed_hurdle = -0.5 * (new_cost * C)  # Speculative hurdle: 50% of detour cost

    if is_mandatory or new_profit >= seed_hurdle:
        # A new route has its best (and only) insertion at pos 0 in a new index
        node_options.append((new_profit, len(routes), 0))

    return node_options


def regret_2_profit_insertion(
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
    Insert removed nodes based on the regret-2 criterion maximizing profit.

    Wrapper around regret_k_profit_insertion with k=2.
    """
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=2,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_k_profit_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Regret-k insertion maximizing profit (revenue - cost).

    Efficiency: O(U * N * L_avg) incremental evaluation using a RoutingCache.
    VRPP logic: Instead of minimizing cost, we calculate profit for each position.
    A node is only considered if its best insertion is profitable or if it's mandatory.
    Regret is calculated as the difference between the best and k-th best profits.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        k: Regret degree.
        mandatory_nodes: List of mandatory node indices.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0) for node in r) for r in routes]

    visited = set()
    for r in routes:
        visited.update(r)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
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
    node_route_cache: Dict[int, List[Tuple[float, int]]] = {}

    def get_best_for_route_profit(node_id: int, r_idx: int) -> Tuple[float, int]:
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
            profit = revenue - cost * C

            # Skip unprofitable positions for non-mandatory nodes before comparison
            if not is_mandatory and profit < -1e-4:
                continue

            if profit > best_r_profit:
                best_r_profit = profit
                best_r_pos = pos
        return best_r_profit, best_r_pos

    def get_seed_profit_regret(node_id: int) -> float:
        node_waste = wastes.get(node_id, 0)
        revenue = node_waste * R
        new_cost = dist_matrix[0, node_id] + dist_matrix[node_id, 0]
        new_profit = revenue - (new_cost * C)
        seed_hurdle = -0.5 * (new_cost * C)
        is_mandatory = node_id in mandatory_nodes_set
        if is_mandatory or new_profit >= seed_hurdle:
            return new_profit
        return -float("inf")

    # Initial population
    for node in unassigned:
        node_route_cache[node] = []
        for i in range(len(routes)):
            p, pos = get_best_for_route_profit(node, i)
            node_route_cache[node].append((p, pos))

    # --- Main Loop ---
    while unassigned:
        all_candidates = []
        skipped_nodes = []

        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            # Options: existing routes + seed
            seed_p = get_seed_profit_regret(node)
            options = [(p, i, pos) for i, (p, pos) in enumerate(node_route_cache[node]) if p != -float("inf")]
            if seed_p != -float("inf"):
                options.append((seed_p, len(routes), 0))

            if not options:
                if not is_mandatory:
                    skipped_nodes.append(node)
                continue

            options.sort(key=lambda x: x[0], reverse=True)

            # Regret: paper §3.2.2 sum formula (sum of differences from best)
            # Note for profit (maximization): regret is (best_profit - jth_profit)
            if len(options) > 1:
                target_k = min(k, len(options))
                regret = sum(options[0][0] - options[j][0] for j in range(1, target_k))
                if len(options) < k:
                    # Priority for nodes with few options
                    regret += (k - len(options)) * 1000.0
            else:
                regret = 1e9

            best_option = options[0]
            # all_candidates record: (regret, node, (profit, r_idx, pos))
            all_candidates.append((regret, node, best_option))

        for out_node in skipped_nodes:
            unassigned.remove(out_node)
            if out_node in node_route_cache:
                del node_route_cache[out_node]

        if not all_candidates:
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                # Expand cache
                new_r_idx = len(routes) - 1
                for u_node in unassigned:
                    p, pos = get_best_for_route_profit(u_node, new_r_idx)
                    node_route_cache[u_node].append((p, pos))
                continue
            else:
                break

        # Maximize regret
        # For profit maximisation: ties broken by highest best profit (equivalent to
        # lowest cost in the minimisation formulation)
        all_candidates.sort(key=lambda x: (x[0], x[2][0]), reverse=True)
        _, best_node, (profit, r_idx, pos) = all_candidates[0]

        if r_idx == len(routes):
            # Seed new route
            routes.append([best_node])
            loads.append(wastes.get(best_node, 0))
            unassigned.remove(best_node)
            del node_route_cache[best_node]
            # Update all nodes with the new route
            new_r_idx = len(routes) - 1
            for u_node in unassigned:
                p, ppos = get_best_for_route_profit(u_node, new_r_idx)
                node_route_cache[u_node].append((p, ppos))
        else:
            # Insert into existing
            routes[r_idx].insert(pos, best_node)
            loads[r_idx] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
            del node_route_cache[best_node]
            # Update modified route for all
            for u_node in unassigned:
                p, ppos = get_best_for_route_profit(u_node, r_idx)
                node_route_cache[u_node][r_idx] = (p, ppos)

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)


def regret_3_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """Convenience wrapper for regret-3 insertion."""
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=3,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_4_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """Convenience wrapper for regret-4 insertion."""
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=4,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_3_profit_insertion(
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
    """Convenience wrapper for regret-3 profit-aware insertion."""
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=3,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_4_profit_insertion(
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
    """Convenience wrapper for regret-4 profit-aware insertion."""
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=4,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )
