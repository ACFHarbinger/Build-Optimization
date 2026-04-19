"""
Type I Unstringing Operator.

Removes node V_i and reconnects the route by reversing two sub-tours.
"""

from typing import Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion._routes import (
    _extract_working_route,
)


def apply_type_i_us(route: List[int], i: int, j: int, k: int) -> List[int]:
    """
    Apply Type I Unstringing move.

    Removes V_i.
    Deletes arcs: (V_{i-1}, V_i), (V_i, V_{i+1}), (V_k, V_{k+1}), (V_j, V_{j+1})
    Inserts arcs: (V_{i-1}, V_k), (V_{i+1}, V_j), (V_{k+1}, V_{j+1})

    Reverses sub-tours (V_{i+1}...V_k) and (V_{k+1}...V_j).

    Args:
        route: The tour as a list of node IDs.
        i: Index of node to remove.
        j: Index of node V_j such that the arc (V_j, V_{j+1}) is broken.
        k: Index of node V_k such that the arc (V_k, V_{k+1}) is broken.

    Returns:
        New route with V_i removed and segments reconnected.
    """
    # Defensive copy logic for route input
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # 1. Rotate the route so V_{i-1} is at index 0.
    pivot = (i - 1) % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    # In rotated route:
    # Index 0: V_{i-1}
    # Index 1: V_i (to be removed)
    # Index 2: V_{i+1}

    val_j = route[j]
    val_k = route[k]

    # Find indices in rotated working route
    try:
        j_new = rot_route.index(val_j)
        k_new = rot_route.index(val_k)
    except ValueError:
        return route  # Fallback

    # Reconnection Logic per Müller & Bonilha (2022):
    # Broken arcs: (v_{i-1}, v_i), (v_i, v_{i+1}), (v_j, v_{j+1}), (v_k, v_{k+1})
    # Reconnect: (v_{i-1}, v_{k+1}), (v_k, v_{j+1}), (v_j, v_{i+1})
    # Segment [v_{i+1}...v_k] is reversed.
    # Segments in rotated route:
    # Index 0: v_{i-1}
    # Index 1: v_i (removed)
    # Index 2: v_{i+1}
    # Indices [2...k_new]: s1 (v_{i+1}...v_k)
    # Indices [k_new+1...j_new]: s2 (v_{k+1}...v_j)
    # Indices [j_new+1...]: remainder (v_{j+1}...v_{i-1})

    s1 = rot_route[2 : k_new + 1]
    s2 = rot_route[k_new + 1 : j_new + 1]
    remainder = rot_route[j_new + 1 :]

    # New sequence: v_{i-1} -> s2_reversed -> s1_reversed -> remainder
    new_rot = [rot_route[0]] + s2[::-1] + s1[::-1] + remainder

    # Rotate back to align with original depot position?
    # Find depot 0 and rotate it to front.
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route


def apply_type_i_us_profit(
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
    Apply Type I Unstringing move and return profit delta.

    Args:
        route: The tour as a list of node IDs.
        i, j, k: Indices as defined in apply_type_i_us.
        dist_matrix: Distance matrix.
        wastes: Waste levels.
        R, C: Revenue and cost multipliers.

    Returns:
        (new_route, delta_profit)
    """
    n, is_closed, work_route, n_work = _extract_working_route(route)

    # Identifiers
    v_i = work_route[i]
    v_im1 = work_route[(i - 1) % n_work]
    v_ip1 = work_route[(i + 1) % n_work]
    v_j = work_route[j]
    v_jp1 = work_route[(j + 1) % n_work]
    v_k = work_route[k]
    v_kp1 = work_route[(k + 1) % n_work]

    # Delta Cost
    # Deletes: (V_{i-1}, V_i), (V_i, V_{i+1}), (V_k, V_{k+1}), (V_j, V_{j+1})
    # Inserts: (V_{i-1}, V_k), (V_{i+1}, V_j), (V_{k+1}, V_{j+1})
    d_del = dist_matrix[v_im1, v_i] + dist_matrix[v_i, v_ip1] + dist_matrix[v_k, v_kp1] + dist_matrix[v_j, v_jp1]
    d_ins = dist_matrix[v_im1, v_k] + dist_matrix[v_ip1, v_j] + dist_matrix[v_kp1, v_jp1]
    delta_cost = d_ins - d_del

    # Delta Revenue
    delta_rev = -wastes.get(v_i, 0.0) * R

    delta_profit = delta_rev - delta_cost * C

    return apply_type_i_us(route, i, j, k), float(delta_profit)
