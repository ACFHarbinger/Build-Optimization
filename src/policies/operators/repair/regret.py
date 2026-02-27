"""
Regret Insertion Operator Module.

This module implements the Regret-k insertion heuristic for VRP repair.
It calculates a regret value for each unassigned node based on the cost difference
between its best and k-th best insertion positions.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.repair.regret import RegretInsertion
    >>> operator = RegretInsertion(k=2)
    >>> new_routes = operator.repair(destroyed_routes, unassigned_nodes)
"""

from typing import Dict, List, Optional

import numpy as np


def regret_2_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: Optional[float] = None,
    mandatory_nodes: Optional[List[int]] = None,
    cost_unit: float = 1.0,
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
        R: Revenue multiplier (Optional).
        mandatory_nodes: Optional list of mandatory node indices.

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
        R=R,
        mandatory_nodes=mandatory_nodes,
        cost_unit=cost_unit,
    )


def regret_k_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    k: int = 2,
    R: Optional[float] = None,
    mandatory_nodes: Optional[List[int]] = None,
    cost_unit: float = 1.0,
) -> List[List[int]]:
    """
    Insert removed nodes using the regret-k heuristic.

    Computes the 'regret' for each unassigned node, defined as the difference
    between the cost of the best insertion and the k-th best insertion. Node
    with the maximum regret is inserted first into its best position.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        k: Regret degree (2, 3, etc.).
        R: Revenue multiplier (Optional).
        mandatory_nodes: Optional list of mandatory node indices.

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

    # All unvisited nodes (including those previously removed) are candidates
    n_nodes = len(dist_matrix) - 1
    unassigned = list(set(range(1, n_nodes + 1)) - visited)

    while unassigned:
        all_candidates = []
        unprofitable_nodes = []

        for node in unassigned:
            waste = wastes.get(node, 0)
            node_options = []

            for r_idx, route in enumerate(routes):
                if loads[r_idx] + waste > capacity:
                    continue

                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                    node_options.append((cost, r_idx, pos))

            # New route option
            new_cost = dist_matrix[0][node] + dist_matrix[node][0]
            node_options.append((new_cost, len(routes), 0))

            # Sort options by cost
            node_options.sort(key=lambda x: x[0])

            # VRPP Logic
            best_cost = node_options[0][0]
            if R is not None:
                revenue = waste * R
                if best_cost * cost_unit > revenue and node not in mandatory_nodes_set:
                    unprofitable_nodes.append(node)
                    continue

            # Calculate regret
            if len(node_options) >= k:
                # Regret = cost_at_k - cost_at_1
                regret = node_options[k - 1][0] - node_options[0][0]
            elif len(node_options) > 1:
                # If fewer than k options, regret is diff between last and first
                regret = node_options[-1][0] - node_options[0][0]
            else:
                # Only one option (or none), max priority
                regret = float("inf")

            best_option = node_options[0] if node_options else (float("inf"), -1, -1)
            all_candidates.append((regret, node, best_option))

        # Remove unprofitable nodes from unassigned
        for node in unprofitable_nodes:
            unassigned.remove(node)

        if not all_candidates:
            # No feasible/profitable insertions left
            # For remaining mandatory nodes, we'll hit this break and need to handle them
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                continue
            else:
                break

        # Pick node with max regret
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        _, best_node, (cost, r_idx, pos) = all_candidates[0]

        if r_idx == -1:
            # Cannot insert node anywhere
            unassigned.remove(best_node)
            continue

        # Apply insertion
        waste = wastes.get(best_node, 0)
        if r_idx == len(routes):
            routes.append([best_node])
            loads.append(waste)
        else:
            routes[r_idx].insert(pos, best_node)
            loads[r_idx] += waste

        unassigned.remove(best_node)

    return routes
