"""
Type IV Unstringing Operator.

Most complex US operator involving four neighbor nodes and multiple reversals.
"""

from typing import Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion._routes import (
    _extract_working_route,
)


def apply_type_iv_us(route: List[int], i: int, j: int, k: int, l: int) -> List[int]:
    """
    Apply Type IV Unstringing move.

    Removes V_i.
    Involves neighbors V_j, V_l, V_k such that in rotated order (starting V_{i+1}):
    V_{i+1} ... V_j ... V_l ... V_k ... V_{i-1}

    Deletes arcs:
    (V_{j-1}, V_j), (V_{l-1}, V_l), (V_k, V_{k+1}), (V_{i-1}, V_i), (V_i, V_{i+1})

    Inserts arcs:
    (V_{j-1}, V_l), (V_k, V_{i-1}), (V_{k+1}, V_{l-1}), (V_j, V_{i+1})

    Reconstructs as:
    S_C + S_D + S_A_rev + S_B_rev
    Where:
    S_C = (V_{i+1}...V_{j-1})
    S_D = (V_l...V_k)
    S_A_rev = (V_{i-1}...V_{k+1})  [Reverse of V_{k+1}...V_{i-1}]
    S_B_rev = (V_{l-1}...V_j)    [Reverse of V_j...V_{l-1}]

    Args:
        route: The tour.
        i: Index of V_i.
        j, k, l: Indices of nodes V_j, V_k, V_l.

    Returns:
        Modified tour.
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # Rotate so V_{i+1} is at index 0.
    # V_i is at index -1 (or n-1).
    pivot = (i + 1) % n_work
    rot_route_full = work_route[pivot:] + work_route[:pivot]

    # Remove V_i (last element of rotated full working route)
    rot_route = rot_route_full[:-1]

    v_j = route[j]
    v_k = route[k]
    v_l = route[l]

    # Find indices in the route (without V_i)
    try:
        j_new = rot_route.index(v_j)
        k_new = rot_route.index(v_k)
        l_new = rot_route.index(v_l)
    except ValueError:
        return route

    s_c = rot_route[:j_new]
    s_b_orig = rot_route[j_new:l_new]
    s_d = rot_route[l_new : k_new + 1]
    s_a_orig = rot_route[k_new + 1 :]

    # Construction: S_C + S_D + S_A_rev + S_B_rev
    new_rot = s_c + s_d + s_a_orig[::-1] + s_b_orig[::-1]

    # Restore depot
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route


def apply_type_iv_us_profit(
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
    Apply Type IV Unstringing and return profit delta.

    Args:
        route: The tour.
        i, j, k, l: Indices as defined in apply_type_iv_us.
        dist_matrix: Distance matrix.
        wastes: Waste levels.
        R, C: Revenue and cost multipliers.

    Returns:
        (new_route, delta_profit)
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # Identifiers
    v_im1 = work_route[(i - 1) % n_work]
    v_i = work_route[i]
    v_ip1 = work_route[(i + 1) % n_work]
    v_jm1 = work_route[(j - 1) % n_work]
    v_j = work_route[j]
    v_lm1 = work_route[(l - 1) % n_work]
    v_l = work_route[l]
    v_k = work_route[k]
    v_kp1 = work_route[(k + 1) % n_work]

    # Delta Cost
    # Deletes: (V_{j-1}, V_j), (V_{l-1}, V_l), (V_k, V_{k+1}), (V_{i-1}, V_i), (V_i, V_{i+1})
    # Inserts: (V_{j-1}, V_l), (V_k, V_{i-1}), (V_{k+1}, V_{l-1}), (V_j, V_{i+1})
    d_del = (
        dist_matrix[v_jm1, v_j]
        + dist_matrix[v_lm1, v_l]
        + dist_matrix[v_k, v_kp1]
        + dist_matrix[v_im1, v_i]
        + dist_matrix[v_i, v_ip1]
    )
    d_ins = dist_matrix[v_jm1, v_l] + dist_matrix[v_k, v_im1] + dist_matrix[v_kp1, v_lm1] + dist_matrix[v_j, v_ip1]
    delta_cost = d_ins - d_del

    # Delta Revenue
    delta_rev = -wastes.get(v_i, 0.0) * R

    delta_profit = delta_rev - delta_cost * C

    return apply_type_iv_us(route, i, j, k, l), float(delta_profit)
