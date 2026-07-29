"""
Type II Unstringing Operator.

Involves three neighbor nodes (V_j, V_k, V_l) and complex reversals.
This is the correct Type II implementation (formerly III).
"""

from typing import Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion._routes import (
    _extract_working_route,
)


def apply_type_ii_us(route: List[int], i: int, j: int, k: int, l: int) -> List[int]:
    """
    Apply Type II Unstringing move.

    Removes V_i.
    Involves neighbors:
    - V_j (neighbor of V_{i+1})
    - V_k (neighbor of V_{i-1})
    - V_l (neighbor of V_{k+1})

    Order in route (relative to i): V_{i+1} ... V_k ... V_j ... V_l ... V_{i-1}

    Reconstructs as:
    V_{i-1} -> S1_rev -> S2_rev -> S3_rev -> Remainder
    Where:
    S1 = (V_{i+1}...V_k)
    S2 = (V_{k+1}...V_j)
    S3 = (V_{j+1}...V_l)

    Inserts arcs: (V_{i-1}, V_k), (V_{i+1}, V_j), (V_{k+1}, V_l), (V_{j+1}, V_{l+1})

    Args:
        route: The tour.
        i: Index of V_i.
        j, k, l: Indices of neighbor nodes conforming to the Type II topology.

    Returns:
        Modified tour.
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    pivot = (i - 1) % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    v_j = route[j]
    v_k = route[k]
    v_l = route[l]

    try:
        j_new = rot_route.index(v_j)
        k_new = rot_route.index(v_k)
        l_new = rot_route.index(v_l)
    except ValueError:
        return route

    # S1: V_{i+1} ... V_k
    s1 = rot_route[2 : k_new + 1]

    # S2: V_{k+1} ... V_j
    s2 = rot_route[k_new + 1 : j_new + 1]

    # S3: V_{j+1} ... V_l
    s3 = rot_route[j_new + 1 : l_new + 1]

    # Remainder: V_{l+1} ...
    remainder = rot_route[l_new + 1 :]

    # Construction
    new_rot = [rot_route[0]] + s1[::-1] + s2[::-1] + s3[::-1] + remainder

    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route


def apply_type_ii_us_profit(
    route: List[int],
    i: int,
    j: int,
    k: int,
    l: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
) -> Tuple[List[int], float]:
    """
    Apply Type II Unstringing and return profit delta.

    Args:
        route: The tour.
        i: Index of V_i.
        j, k, l: Indices as defined in apply_type_ii_us.
        dist_matrix: Distance matrix.
        wastes: Waste levels.
        R, C: Revenue and cost multipliers.

    Returns:
        (new_route, delta_profit)
    """
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

    # Identifiers
    v_im1 = work_route[(i - 1) % n_work]
    v_i = work_route[i]
    v_ip1 = work_route[(i + 1) % n_work]
    v_j = work_route[j]
    v_jp1 = work_route[(j + 1) % n_work]
    v_k = work_route[k]
    v_kp1 = work_route[(k + 1) % n_work]
    v_l = work_route[l]
    v_lp1 = work_route[(l + 1) % n_work]

    # Delta Cost
    # Deletes: (V_{i-1}, V_i), (V_i, V_{i+1}), (V_k, V_{k+1}), (V_j, V_{j+1}), (V_l, V_{l+1})
    # Inserts: (V_{i-1}, V_k), (V_{i+1}, V_j), (V_{k+1}, V_l), (V_{j+1}, V_{l+1})
    d_del = (
        dist_matrix[v_im1, v_i]
        + dist_matrix[v_i, v_ip1]
        + dist_matrix[v_k, v_kp1]
        + dist_matrix[v_j, v_jp1]
        + dist_matrix[v_l, v_lp1]
    )
    d_ins = dist_matrix[v_im1, v_k] + dist_matrix[v_ip1, v_j] + dist_matrix[v_kp1, v_l] + dist_matrix[v_jp1, v_lp1]
    delta_cost = d_ins - d_del

    # Delta Revenue
    delta_rev = -wastes.get(v_i, 0.0) * R

    delta_profit = delta_rev - delta_cost * C

    return apply_type_ii_us(route, i, j, k, l), float(delta_profit)
