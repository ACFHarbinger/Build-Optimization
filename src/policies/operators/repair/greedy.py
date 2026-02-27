"""
Greedy Insertion Operator Module.

This module implements the greedy insertion heuristic, which iteratively inserts
unassigned nodes into the position that minimizes the immediate cost increase.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.repair.greedy import greedy_insertion
    >>> routes = greedy_insertion(routes, removed, dist_matrix, wastes, capacity)
"""

from typing import Dict, List, Optional

import numpy as np


def greedy_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: Optional[float] = None,
    mandatory_nodes: Optional[List[int]] = None,
    cost_unit: float = 1.0,
    expand_pool: bool = True,
) -> List[List[int]]:
    """
    Insert removed nodes into their best (cheapest) positions greedily.

    Iterates through all unassigned nodes and all possible insertion positions,
    finding the globally cheapest insertion and applying it. Repeats until all
    nodes are inserted OR skipping occurs based on profitability (VRPP).

    Args:
        routes: Partial routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier (Optional). If provided, insertion is skipped if cost > revenue.
        mandatory_nodes: List of mandatory node indices.
        cost_unit: Cost per distance unit.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    # Calculate current loads and track visited nodes
    loads = []
    visited = set()
    for route in routes:
        loads.append(sum(wastes.get(node, 0) for node in route))
        visited.update(route)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
        n_nodes = len(dist_matrix) - 1
        unassigned = list(set(range(1, n_nodes + 1)) - visited)
    else:
        unassigned = list(removed_nodes)

    while unassigned:
        best_cost = float("inf")
        best_node = -1
        best_route_idx = -1
        best_pos = -1

        for node in unassigned:
            node_waste = wastes.get(node, 0)
            revenue = node_waste * R if R is not None else float("inf")
            is_mandatory = node in mandatory_nodes_set

            for i, route in enumerate(routes):
                if loads[i] + node_waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    # Cost increase: d(i-1, node) + d(node, i) - d(i-1, i)
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0

                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

                    if cost < best_cost:
                        # VRPP check: skip if cost * cost_unit > revenue and not mandatory
                        if R is not None and cost * cost_unit > revenue and not is_mandatory:
                            continue

                        best_cost = cost
                        best_node = node
                        best_route_idx = i
                        best_pos = pos

        if best_node != -1:
            routes[best_route_idx].insert(best_pos, best_node)
            loads[best_route_idx] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
        else:
            # If no feasible insertions are found, we must handle any remaining mandatory nodes
            # by creating new routes if necessary, but ALNS usually relies on destroy/repair loops.
            # For VRPP, we allow skipping non-mandatory nodes.
            # If there are remaining mandatory nodes, we should probably try to create new routes.
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
            else:
                # No more mandatory nodes and no more profitable/feasible insertions
                break

    return routes
