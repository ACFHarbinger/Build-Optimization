"""
Type II Unstringing Operator.

Removes node V_i and reconnects the route involving V_j and V_k where k > j.
"""

from typing import List


def apply_type_ii_unstringing(route: List[int], i: int, j: int, k: int) -> List[int]:
    """
    Apply Type II Unstringing move.

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
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

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
