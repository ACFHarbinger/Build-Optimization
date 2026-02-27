"""
Type III Unstringing Operator.

Involves three neighbor nodes (V_j, V_k, V_l) and complex reversals.
"""

from typing import List


def apply_type_iii_unstringing(route: List[int], i: int, j: int, k: int, l: int) -> List[int]:
    """
    Apply Type III Unstringing move.

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
        j, k, l: Indices of neighbor nodes conforming to the Type III topology.

    Returns:
        Modified tour.
    """
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

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

    # Validation of order? We assume the move generator provides valid indices.
    # Expected: 2 <= k_new < j_new < l_new < n

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
