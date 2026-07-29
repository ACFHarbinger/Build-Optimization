"""
Deep Insertion Operator Module.

An intensive variant of greedy insertion that scores each candidate position
not only by the immediate cost delta but also by the residual capacity
utility of the receiving route.  This favours balanced loads and avoids
overloading routes with high residual capacity.

Score formula:
    score = cost_delta - alpha * residual_capacity_utility

where ``residual_capacity_utility = (capacity - load - node_demand) / capacity``.

A lower ``alpha`` behaves like standard greedy; higher ``alpha`` biases
toward more balanced solutions. This balancing utility is inspired by the
load-balancing variants in Archetti, Speranza, and Vigo (2014), adapted as
an intensive ("deep") repair operator for ALNS.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.deep import deep_insertion
    >>> routes = deep_insertion(routes, removed, dist_matrix, wastes, capacity, alpha=0.3)
"""

from typing import Dict, List, Optional

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def deep_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    alpha: float = 0.3,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Deep insertion: cost-delta minus capacity-utility scoring.

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: Waste/demand look-up per node.
        capacity: Vehicle capacity.
        alpha: Weight for residual capacity penalty (>= 0).
        mandatory_nodes: List of nodes that must be inserted.
        expand_pool: If True, evaluates all unvisited nodes.

    Returns:
        Updated routes with nodes inserted.

    Note:
        Deep Insertion (Archetti et al. 2014 adaptation):
        1. Select node *v* by greedy order (prioritizing mandatory).
        2. Score each position *p* by its marginal cost *d(p)* adjusted by
           the route's residual capacity utility *U(r)*.
        3. Insert *v* at the position that minimizes ``d(p) - alpha * U(r)``.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    # Prioritize mandatory nodes by placing them at the front
    unassigned.sort(key=lambda x: 0 if x in mandatory_set else 1)
    for node in unassigned:
        node_waste = wastes.get(node, 0.0)
        is_mandatory = node in mandatory_set

        best_score = float("inf")
        best_route_idx = -1
        best_pos = -1

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            residual = (capacity - loads[r_idx] - node_waste) / capacity

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                score = cost_delta - alpha * residual

                if score < best_score:
                    best_score = score
                    best_route_idx = r_idx
                    best_pos = pos

        # Check new route
        new_route_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_residual = (capacity - node_waste) / capacity
        new_score = new_route_cost - alpha * new_residual

        if new_score < best_score:
            best_score = new_score
            best_route_idx = len(routes)
            best_pos = 0

        # Execute Best Move
        if best_route_idx != -1:
            if best_route_idx == len(routes):
                routes.append([node])
                loads.append(node_waste)
            else:
                routes[best_route_idx].insert(best_pos, node)
                loads[best_route_idx] += node_waste
        elif is_mandatory:
            # Safety fallback for mandatory nodes
            routes.append([node])
            loads.append(node_waste)

    return routes


def deep_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    alpha: float = 0.3,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Deep profit-driven insertion: profit plus capacity-utility scoring.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        alpha: Weight for residual capacity bonus (>= 0).
        mandatory_nodes: List of mandatory node indices.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    while unassigned:
        best_score = -float("inf")
        best_node = -1
        best_route = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0)
            revenue = node_waste * R
            is_mandatory = node in mandatory_nodes_set

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + node_waste > capacity:
                    continue

                residual = (capacity - loads[r_idx] - node_waste) / max(capacity, 1e-9)

                for pos in range(len(route) + 1):
                    # Cost increase: d(prev, node) + d(node, nxt) - d(prev, nxt)
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    cost_delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    profit = revenue - (cost_delta * C)

                    # Score = Profit + Alpha * Residual Utility
                    score = profit + alpha * residual
                    effective_score = score + (1e9 if is_mandatory else 0)

                    if effective_score > best_score:
                        if not is_mandatory and profit < -1e-4:
                            continue
                        best_score = effective_score
                        best_node = node
                        best_route = r_idx
                        best_pos = pos

            # Evaluate new route (Speculative Seeding)
            new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
            new_profit = revenue - (new_cost * C)
            seed_hurdle = -0.5 * (new_cost * C)

            if is_mandatory or new_profit >= seed_hurdle:
                # Score = Profit + Alpha * (Capacity Utility)
                # For a new route, residual = (capacity - node_waste) / capacity
                new_residual = (capacity - node_waste) / max(capacity, 1e-9)
                new_score = new_profit + alpha * new_residual
                new_effective_score = new_score + (1e9 if is_mandatory else 0)

                if new_effective_score > best_score:
                    best_score = new_effective_score
                    best_node = node
                    best_route = len(routes)
                    best_pos = 0

        if best_node != -1:
            if best_route == len(routes):
                routes.append([best_node])
                loads.append(node_waste)
            else:
                routes[best_route].insert(best_pos, best_node)
                loads[best_route] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
        else:
            # Handle remaining mandatory nodes
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                continue
            else:
                break

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
