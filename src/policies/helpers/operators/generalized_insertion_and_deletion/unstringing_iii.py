"""
Type III Unstringing Operator.

Removes node V_i and reconnects the route involving V_j and V_k where k > j.
This is the correct Type III implementation (formerly II).
"""

from typing import Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion._routes import (
    _extract_working_route,
)


def apply_type_iii_us(route: List[int], i: int, j: int, k: int) -> List[int]:
    """
    Apply Type III Unstringing move.

    Removes V_i.
    Reverses sub-tours (V_{i+1}...V_j) and (V_{j+1}...V_k).
    Reconnects sequence: V_{i-1} -> V_k...V_{j+1} -> V_j...V_{i+1} -> V_{k+1}...

    Args:
        route: The tour as a list of node IDs.
        i: Index of node V_i to remove.
        j: Index of node V_j (neighbor of V_{i+1}).
        k: Index of node V_k (neighbor of V_{i-1}), where k > j relative to i.

    Returns:
        The modified tour.
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # Rotate so V_{i-1} is at index 0, V_i is at 1.
    pivot = (i - 1) % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    # In rotated route:
    v_j_val = route[j]
    v_k_val = route[k]

    # Find new indices
    try:
        j_new = rot_route.index(v_j_val)
        k_new = rot_route.index(v_k_val)
    except ValueError:
        return route

    # Segments:
    # S1: V_{i+1} ... V_j -> indices [2, j_new + 1)
    s1 = rot_route[2 : j_new + 1]

    # S2: V_{j+1} ... V_k -> indices [j_new + 1, k_new + 1)
    s2 = rot_route[j_new + 1 : k_new + 1]

    # Remainder: V_{k+1} ... -> indices [k_new + 1, end]
    remainder = rot_route[k_new + 1 :]

    # Construction:
    # V_{i-1} -> S2_rev -> S1_rev -> Remainder
    new_rot = [rot_route[0]] + s2[::-1] + s1[::-1] + remainder

    # Restore depot
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route


def apply_type_iii_us_profit(
    route: List[int],
    i: int,
    j: int,
    k: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
) -> Tuple[List[int], float]:
    """
    Apply Type III Unstringing and return profit delta.

    Args:
        route: The tour as a list of node IDs.
        i, j, k: Indices as defined in apply_type_iii_us.
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
    v_j = work_route[j]
    v_jp1 = work_route[(j + 1) % n_work]
    v_k = work_route[k]
    v_kp1 = work_route[(k + 1) % n_work]

    # Delta Cost
    # Deletes: (V_{i-1}, V_i), (V_i, V_{i+1}), (V_j, V_{j+1}), (V_k, V_{k+1})
    # Inserts: (V_{i-1}, V_k), (V_{j+1}, V_j), (V_{i+1}, V_{k+1})
    d_del = dist_matrix[v_im1, v_i] + dist_matrix[v_i, v_ip1] + dist_matrix[v_j, v_jp1] + dist_matrix[v_k, v_kp1]
    d_ins = dist_matrix[v_im1, v_k] + dist_matrix[v_jp1, v_j] + dist_matrix[v_ip1, v_kp1]
    delta_cost = d_ins - d_del

    # Delta Revenue
    delta_rev = -wastes.get(v_i, 0.0) * R

    delta_profit = delta_rev - delta_cost * C

    return apply_type_iii_us(route, i, j, k), float(delta_profit)
