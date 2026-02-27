"""
Type IV Unstringing Operator.

Most complex US operator involving four neighbor nodes and multiple reversals.
"""

from typing import List


def apply_type_iv_unstringing(route: List[int], i: int, j: int, k: int, l: int) -> List[int]:
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
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

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
