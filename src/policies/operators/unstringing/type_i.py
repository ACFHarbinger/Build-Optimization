"""
Type I Unstringing Operator.

Removes node V_i and reconnects the route by reversing two sub-tours.
"""

from typing import List


def apply_type_i_unstringing(route: List[int], i: int, j: int, k: int) -> List[int]:
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
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

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

    # Segments:
    # S1: V_{i+1}...V_k -> indices [2, k_new + 1)
    s1 = rot_route[2 : k_new + 1]

    # S2: V_{k+1}...V_j -> indices [k_new + 1, j_new + 1)
    s2 = rot_route[k_new + 1 : j_new + 1]

    # Remainder: V_{j+1}...V_{i-1} (start) -> indices [j_new + 1, end]
    remainder = rot_route[j_new + 1 :]

    # Reconnection Logic:
    new_rot = [rot_route[0]] + s1[::-1] + s2[::-1] + remainder

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
