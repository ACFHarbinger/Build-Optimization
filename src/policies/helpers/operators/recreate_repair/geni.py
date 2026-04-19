"""
GENI (Generalized Insertion) Operator Module.

This module implements the exact GENI Type I and Type II constructive insertion
moves as defined by Gendreau, Hertz, and Laporte (1992). It inserts a node `u`
between two *non-adjacent* nodes in the tour while optimally reconnecting and
reversing the intermediate segments.

It contains:
1. `geni_insertion`: The standard, distance-minimizing operator for CVRP.
2. `geni_profit_insertion`: A profit-aware VRPP variant utilizing speculative
   seeding and economic pruning.

Algorithm:
    The Generalized Insertion (GENI) procedure (Gendreau et al. 1992) selects
    insertion positions based on two complex structural move types:
    * **Type I**: A 3-edge exchange that deletes (v_i, v_{i+1}), (v_j, v_{j+1}),
      (v_k, v_{k+1}) and reconnects segments to insert node *u*.
    * **Type II**: A 4-edge exchange that deletes (v_i, v_{i+1}), (v_{l-1}, v_l),
      (v_j, v_{j+1}), (v_{k-1}, v_k) and reverses intermediate paths to optimally
      place *u*.
    The p-neighborhood search facilitates efficient evaluation of high-quality
    insertion positions.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.geni import geni_insertion
    >>> improved = geni_insertion(ls, node=5, r_idx=0, neighborhood_size=5)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.utils.policy.neighborhood import (
    get_p_neighborhood,
)
from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def _get_rev_cost(forward_cost: np.ndarray, backward_cost: np.ndarray, start: int, end: int) -> float:
    """Calculates the cost difference if the segment full_route[start:end] is reversed in O(1)."""
    if start >= end - 1:
        return 0.0
    # old_cost: sum(dist_matrix[full[k], full[k+1]]) from k=start to end-2
    # new_cost: sum(dist_matrix[full[k+1], full[k]]) from k=start to end-2
    old_cost = forward_cost[end - 1] - forward_cost[start]
    new_cost = backward_cost[end - 1] - backward_cost[start]
    return new_cost - old_cost


def _apply_geni_move(route: List[int], u: int, i: int, j: int, k: int, l: int, m_type: str) -> List[int]:
    """Applies the exact structural reconnections for GENI insertions."""
    if m_type == "SIMPLE":
        return route[:i] + [u] + route[i:]

    full = [0] + route + [0]

    if m_type == "TYPE_I":
        # Deletes (v_i, v_{i+1}), (v_j, v_{j+1}), (v_k, v_{k+1})
        # Sequence: P1 + [u] + P2_rev + P3_rev + P4
        new_full = full[: i + 1] + [u] + full[i + 1 : j + 1][::-1] + full[j + 1 : k + 1][::-1] + full[k + 1 :]
    elif m_type == "TYPE_II":
        # Phase 2: Correct Type II path (exactly two reversals)
        # Deletes (v_i, v_{i+1}), (v_{l-1}, v_l), (v_j, v_{j+1}), (v_{k-1}, v_k)
        # Sequence: P1 + [u] + P3A_rev (v_j..v_l) + P3B (v_{j+1}..v_{k-1}) + P2_rev (v_{l-1}..v_{i+1}) + P4 (v_k..)
        new_full = full[: i + 1] + [u] + full[l : j + 1][::-1] + full[j + 1 : k] + full[i + 1 : l][::-1] + full[k:]
    else:
        new_full = full

    return new_full[1:-1]


def _evaluate_route(  # noqa: C901
    u: int,
    route: List[int],
    dist_matrix: np.ndarray,
    neighborhood_size: int,
    revenue: Optional[float],
    C: float,
    is_man: bool,
    use_deterministic_p_neighborhood: bool = False,
) -> Tuple[float, Optional[Tuple[int, int, int, int, str]]]:
    """Evaluates all GENI moves for a specific route."""
    is_profit = revenue is not None
    best_val = -float("inf") if is_profit else float("inf")
    best_move = None

    full = [0] + route + [0]
    n_full = len(full)

    # Phase 4: Precompute prefix sums for O(1) reversal cost
    # forward_cost[x] = sum(dist(full[k], full[k+1])) for k in 0..x-1
    # backward_cost[x] = sum(dist(full[k+1], full[k])) for k in 0..x-1
    forward_cost = np.zeros(n_full)
    backward_cost = np.zeros(n_full)
    for k in range(n_full - 1):
        forward_cost[k + 1] = forward_cost[k] + dist_matrix[full[k], full[k + 1]]
        backward_cost[k + 1] = backward_cost[k] + dist_matrix[full[k + 1], full[k]]

    # Map nodes to indices once
    node_to_indices: Dict[int, List[int]] = {}
    for idx, node in enumerate(full):
        node_to_indices.setdefault(node, []).append(idx)

    if use_deterministic_p_neighborhood:
        p_neighbors_u = get_p_neighborhood(u, full[:-1], dist_matrix, neighborhood_size)
        candidate_i = []
        for node in p_neighbors_u:
            candidate_i.extend(node_to_indices.get(node, []))
        candidate_i = [idx for idx in candidate_i if idx < n_full - 1]
    else:
        if neighborhood_size > 0 and (n_full - 1) > neighborhood_size:
            route_nodes = np.array(full[:-1])
            candidate_i = np.argsort(dist_matrix[route_nodes, u])[:neighborhood_size].tolist()
        else:
            candidate_i = list(range(n_full - 1))

    for i in candidate_i:
        # 1. SIMPLE
        delta = dist_matrix[full[i], u] + dist_matrix[u, full[i + 1]] - dist_matrix[full[i], full[i + 1]]
        val = (revenue - delta * C) if is_profit else delta
        if (is_profit and val > best_val and (is_man or val >= -1e-4)) or (not is_profit and val < best_val):
            best_val, best_move = val, (i, -1, -1, -1, "SIMPLE")

        if n_full < 4:
            continue

        # Get candidates for v_j in N_p(u)
        candidate_j = [idx for idx in candidate_i if idx > i + 1]

        for j in candidate_j:
            # 2. TYPE I (3-opt)
            v_ip1 = full[i + 1]
            p_neighbors_ip1 = get_p_neighborhood(v_ip1, full, dist_matrix, neighborhood_size)
            candidate_k_i = []
            for node in p_neighbors_ip1:
                candidate_k_i.extend(node_to_indices.get(node, []))
            candidate_k_i = [idx for idx in candidate_k_i if idx > j and idx < n_full - 1]

            for k in candidate_k_i:
                rev_cost_s1 = _get_rev_cost(forward_cost, backward_cost, i + 1, j + 1)
                rev_cost_s2 = _get_rev_cost(forward_cost, backward_cost, j + 1, k + 1)

                delta_i = (
                    dist_matrix[full[i], u]
                    + dist_matrix[u, full[j]]
                    + dist_matrix[full[i + 1], full[k]]
                    + dist_matrix[full[j + 1], full[k + 1]]
                    - dist_matrix[full[i], full[i + 1]]
                    - dist_matrix[full[j], full[j + 1]]
                    - dist_matrix[full[k], full[k + 1]]
                    + rev_cost_s1
                    + rev_cost_s2
                )
                val_i = (revenue - delta_i * C) if is_profit else delta_i
                if (is_profit and val_i > best_val and (is_man or val_i >= -1e-4)) or (
                    not is_profit and val_i < best_val
                ):
                    best_val, best_move = val_i, (i, j, k, -1, "TYPE_I")

            # 3. TYPE II (4-opt)
            # ordering: i < l < j < k
            candidate_l = [idx for idx in range(i + 2, j)]
            for l in candidate_l:
                # v_k in N_p(v_{i+1})
                v_ip1 = full[i + 1]
                p_neighbors_ip1 = get_p_neighborhood(v_ip1, full, dist_matrix, neighborhood_size)
                candidate_k_ii = []
                for node in p_neighbors_ip1:
                    candidate_k_ii.extend(node_to_indices.get(node, []))
                candidate_k_ii = [idx for idx in candidate_k_ii if idx > j and idx < n_full - 1]

                for k in candidate_k_ii:
                    # v_l in N_p(v_{j+1})
                    # Phase 3 logic also applied here for consistency
                    v_jp1 = full[j + 1]
                    p_neighbors_jp1 = get_p_neighborhood(v_jp1, full, dist_matrix, neighborhood_size)
                    if full[l] not in p_neighbors_jp1:
                        continue

                    rev_cost_s2 = _get_rev_cost(forward_cost, backward_cost, i + 1, l)  # Reverse v_{i+1}...v_{l-1}
                    rev_cost_s3a = _get_rev_cost(forward_cost, backward_cost, l, j + 1)  # Reverse v_l...v_j

                    # Phase 2 Corrections:
                    # Deletions: (v_i, v_{i+1}), (v_{l-1}, v_l), (v_j, v_{j+1}), (v_{k-1}, v_k)
                    # Insertions: (v_i, u), (u, v_j), (v_l, v_{j+1}), (v_{k-1}, v_{l-1}), (v_{i+1}, v_k)
                    delta_ii = (
                        dist_matrix[full[i], u]
                        + dist_matrix[u, full[j]]
                        + dist_matrix[full[l], full[j + 1]]
                        + dist_matrix[full[k - 1], full[l - 1]]
                        + dist_matrix[full[i + 1], full[k]]
                        - dist_matrix[full[i], full[i + 1]]
                        - dist_matrix[full[l - 1], full[l]]
                        - dist_matrix[full[j], full[j + 1]]
                        - dist_matrix[full[k - 1], full[k]]
                        + rev_cost_s2
                        + rev_cost_s3a
                    )
                    val_ii = (revenue - delta_ii * C) if is_profit else delta_ii
                    if (is_profit and val_ii > best_val and (is_man or val_ii >= -1e-4)) or (
                        not is_profit and val_ii < best_val
                    ):
                        best_val, best_move = val_ii, (i, j, k, l, "TYPE_II")

    return best_val, best_move


def _find_best_geni_move(
    u: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    u_waste: float,
    capacity: float,
    neighborhood_size: int,
    revenue: Optional[float] = None,
    C: float = 1.0,
    mandatory_set: Optional[Set[int]] = None,
    use_deterministic_p_neighborhood: bool = False,
) -> Tuple[float, Optional[Tuple[int, int, int, int, int, str, bool]]]:
    """Finds best GENI move for node u across all routes (bidirectional)."""
    is_profit = revenue is not None
    best_val = -float("inf") if is_profit else float("inf")
    best_move = None
    is_man = u in (mandatory_set or set())

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + u_waste > capacity:
            continue

        # Evaluate Forward
        val_f, move_f = _evaluate_route(
            u, route, dist_matrix, neighborhood_size, revenue, C, is_man, use_deterministic_p_neighborhood
        )
        if move_f and ((is_profit and val_f > best_val) or (not is_profit and val_f < best_val)):
            best_val = val_f
            best_move = (r_idx, move_f[0], move_f[1], move_f[2], move_f[3], move_f[4], False)

        # Evaluate Backward
        val_b, move_b = _evaluate_route(
            u, route[::-1], dist_matrix, neighborhood_size, revenue, C, is_man, use_deterministic_p_neighborhood
        )
        if move_b and ((is_profit and val_b > best_val) or (not is_profit and val_b < best_val)):
            best_val = val_b
            best_move = (r_idx, move_b[0], move_b[1], move_b[2], move_b[3], move_b[4], True)

    return best_val, best_move


def geni_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    neighborhood_size: int = 5,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
    expand_pool: bool = False,
    use_deterministic_p_neighborhood: bool = False,
) -> List[List[int]]:
    """
    Standard GENI insertion. Inserts nodes to strictly minimize total distance
    using Simple, Type I, and Type II topology bypasses.

    Args:
        routes: List of active routes.
        removed_nodes: Nodes needing re-insertion.
        dist_matrix: Network distance matrix.
        wastes: Dictionary of node demands.
        capacity: Max load per vehicle.
        neighborhood_size: Restricts the search for v_i to the k-nearest nodes to u in the route.
        mandatory_nodes: List of mandatory nodes ensuring safety fallback insertions.
        rng: Random number generator.
        expand_pool: Whether to expand the pool of candidate nodes.
        use_deterministic_p_neighborhood: If True, use strict p-neighborhood from GHL 1992.

    Returns:
        List[List[int]]: Updated routes.

    Note:
        Implements Generalized Insertion (Gendreau et al. 1992):
        1. For each node *u*, evaluate all Type I and Type II moves across
           candidate routes.
        2. Type I (GHL 1992 Fig 1): 3-edge swap inserting *u* between v_i and v_j.
        3. Type II (GHL 1992 Fig 2): 4-edge swap inserting *u* between v_i and v_j
           with path reversals.
        4. Complexity is controlled via a p-neighborhood restricting candidates
           in each route to the *p* nodes closest to *u*.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]
    if rng is None:
        rng = Random()

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = list(removed_nodes)

    rng.shuffle(unassigned)
    for u in unassigned:
        u_waste = wastes.get(u, 0.0)
        best_delta, best_move = _find_best_geni_move(
            u,
            routes,
            loads,
            dist_matrix,
            u_waste,
            capacity,
            neighborhood_size,
            mandatory_set=mandatory_set,
            use_deterministic_p_neighborhood=use_deterministic_p_neighborhood,
        )

        new_cost = dist_matrix[0, u] + dist_matrix[u, 0]
        if new_cost < best_delta:
            best_delta, best_move = new_cost, (len(routes), 0, 0, 0, 0, "NEW", False)

        if best_move:
            r_idx, i, j, k, l, m_type, is_rev = best_move
            if m_type == "NEW":
                routes.append([u])
                loads.append(u_waste)
            else:
                target_route = routes[r_idx][::-1] if is_rev else routes[r_idx]
                routes[r_idx] = _apply_geni_move(target_route, u, i, j, k, l, m_type)
                loads[r_idx] += u_waste
        elif u in mandatory_set:
            routes.append([u])
            loads.append(u_waste)

    return routes


def geni_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_size: int = 5,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    rng: Optional[Random] = None,
    use_deterministic_p_neighborhood: bool = False,
) -> List[List[int]]:
    """Profit-aware VRPP GENI insertion with speculative seeding."""
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]
    if rng is None:
        rng = Random()

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = list(removed_nodes)

    rng.shuffle(unassigned)
    for u in unassigned:
        u_waste = wastes.get(u, 0.0)
        revenue = u_waste * R
        best_profit, best_move = _find_best_geni_move(
            u,
            routes,
            loads,
            dist_matrix,
            u_waste,
            capacity,
            neighborhood_size,
            revenue,
            C,
            mandatory_set,
            use_deterministic_p_neighborhood,
        )

        new_cost_c = (dist_matrix[0, u] + dist_matrix[u, 0]) * C
        new_profit = revenue - new_cost_c
        seed_hurdle = -0.5 * new_cost_c

        if new_profit > best_profit and (u in mandatory_set or new_profit >= seed_hurdle):
            best_profit, best_move = new_profit, (len(routes), 0, 0, 0, 0, "NEW", False)

        if best_move:
            r_idx, i, j, k, l, m_type, is_rev = best_move
            if m_type == "NEW":
                routes.append([u])
                loads.append(u_waste)
            else:
                target_route = routes[r_idx][::-1] if is_rev else routes[r_idx]
                routes[r_idx] = _apply_geni_move(target_route, u, i, j, k, l, m_type)
                loads[r_idx] += u_waste
        elif u in mandatory_set:
            routes.append([u])
            loads.append(u_waste)

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
