"""
Type II Stringing Operator.

Inserts node V_x back into the route with more complex reconnection.
This is the inverse operation of Type II Unstringing.
"""

from typing import Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion._routes import (
    _extract_working_route,
)


def apply_type_ii_s(route: List[int], x: int, i: int, j: int, k: int, l: int, current_load: float) -> List[int]:
    """
    Apply Type II Stringing move.

    Inserts V_x between V_i and V_j.
    Deletes arcs: (V_i, V_{i+1}), (V_{l-1}, V_l), (V_j, V_{j+1}), (V_{k-1}, V_k)
    Inserts arcs: (V_i, V_x), (V_x, V_j), (V_l, V_{j+1}), (V_{k-1}, V_{l-1}), (V_{i+1}, V_k)

    Reverses sub-tours (V_{i+1}...V_{l-1}) and (V_l...V_j).

    Args:
        route: The tour as a list of node IDs.
        x: Node ID to insert (V_x).
        i: Index of node V_i (before insertion point).
        j: Index of node V_j (after insertion point).
        k: Index of node V_k (reconnection point).
        l: Index of node V_l (reconnection point).
        current_load: Pre-calculated current weight of the route.

    Returns:
        New route with V_x inserted and segments reconnected.

    Constraints:
        V_k != V_j, V_k != V_{j+1}, V_l != V_i, V_l != V_{i+1}
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # Validate constraints
    val_k = route[k]
    val_j = route[j]
    val_l = route[l]
    val_i = route[i]

    if val_k == val_j or val_k == route[(j + 1) % n_work]:
        return route
    if val_l == val_i or val_l == route[(i + 1) % n_work]:
        return route

    # Rotate so V_i is at index 0
    pivot = i % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    # Find indices in rotated route
    try:
        j_new = rot_route.index(val_j)
        k_new = rot_route.index(val_k)
        l_new = rot_route.index(val_l)
    except ValueError:
        return route

    # Segments (assuming i < l < j < k in circular order):
    # S1: V_{i+1}...V_{l-1} -> indices [1, l_new)
    s1 = rot_route[1:l_new]

    # S2: V_l...V_j -> indices [l_new, j_new + 1)
    s2 = rot_route[l_new : j_new + 1]

    # S3: V_{j+1}...V_{k-1} -> indices [j_new + 1, k_new)
    s3 = rot_route[j_new + 1 : k_new]

    # Remainder: V_k...end -> indices [k_new, end]
    remainder = rot_route[k_new:]

    # Reconnection Logic:
    # V_i -> V_x -> V_j (reversed S2) -> V_{j+1} (S3) -> V_l
    # -> V_{l-1} (reversed S1) -> V_k
    # Sequence: V_i -> V_x -> s2_rev -> s3 -> s1_rev -> remainder
    new_rot = [rot_route[0]] + [x] + s2[::-1] + s3 + s1[::-1] + remainder

    # Restore depot
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route


def apply_type_ii_s_profit(
    route: List[int],
    x: int,
    i: int,
    j: int,
    k: int,
    l: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    current_load: float,
    capacity: float,
    R: float,
    C: float,
) -> Tuple[List[int], float]:
    """
    Apply Type II Stringing and return profit delta.

    Args:
        route: The tour as a list of node IDs.
        x: Node ID to insert.
        i, j, k, l: Indices as defined in apply_type_ii_s.
        dist_matrix: Distance matrix.
        wastes: Waste levels.
        current_load: Pre-calculated current weight of the route.
        capacity: Vehicle capacity.
        R, C: Revenue and cost multipliers.

    Returns:
        (new_route, delta_profit)
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # Identifiers
    v_i = work_route[i]
    v_ip1 = work_route[(i + 1) % n_work]
    v_lm1 = work_route[(l - 1) % n_work]
    v_l = work_route[l]
    v_j = work_route[j]
    v_jp1 = work_route[(j + 1) % n_work]
    v_km1 = work_route[(k - 1) % n_work]
    v_k = work_route[k]

    # Constraints
    if v_k in (v_j, work_route[(j + 1) % n_work]):
        return route, -float("inf")
    if v_l in (v_i, work_route[(i + 1) % n_work]):
        return route, -float("inf")

    # Feasibility
    node_waste = wastes.get(x, 0.0)
    if current_load + node_waste > capacity:
        return route, -float("inf")

    # Delta Cost
    # Deletes: (V_i, V_{i+1}), (V_{l-1}, V_l), (V_j, V_{j+1}), (V_{k-1}, V_k)
    # Inserts: (V_i, V_x), (V_x, V_j), (V_l, V_{j+1}), (V_{k-1}, V_{l-1}), (V_{i+1}, V_k)
    d_del = dist_matrix[v_i, v_ip1] + dist_matrix[v_lm1, v_l] + dist_matrix[v_j, v_jp1] + dist_matrix[v_km1, v_k]
    d_ins = (
        dist_matrix[v_i, x]
        + dist_matrix[x, v_j]
        + dist_matrix[v_l, v_jp1]
        + dist_matrix[v_km1, v_lm1]
        + dist_matrix[v_ip1, v_k]
    )
    delta_cost = d_ins - d_del

    # Delta Revenue
    delta_rev = node_waste * R

    delta_profit = delta_rev - delta_cost * C

    return apply_type_ii_s(route, x, i, j, k, l, current_load), float(delta_profit)
