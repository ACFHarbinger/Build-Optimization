"""
3-opt Intra-Route Operator Module.

This module implements the 3-opt intra-route operator, which removes three edges
from a route and reconnects the resulting segments in a new order to reduce
the total tour length.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.route.three_opt_intra import move_3opt_intra
    >>> improved = move_3opt_intra(ls, u, v, r_u, p_u, r_v, p_v)
"""

import random


def move_3opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    3-opt intra-route operator: reconnect three segments within a route.

    Randomly selects a third cut point and evaluates all 3-opt reconnection
    patterns. Applies the best improving move if found.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First cut point node.
        v: Second cut point node.
        r_u: Index of the route.
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.

    Returns:
        bool: True if a 3-opt move was applied (improving), False otherwise.
    """
    route = ls.routes[r_u]
    if len(route) < 4:
        return False
    if p_u > p_v:
        p_u, p_v = p_v, p_u
        u, v = v, u

    for _ in range(5):
        p_w = random.randint(0, len(route) - 1)
        if p_w in {p_u, p_v, p_u + 1, p_v + 1, p_u - 1, p_v - 1}:
            continue

        idx = sorted([p_u, p_v, p_w])
        i, j, k = idx[0], idx[1], idx[2]

        A, B = route[i], route[i + 1] if i < len(route) - 1 else 0
        C, D = route[j], route[j + 1] if j < len(route) - 1 else 0
        E, F = route[k], route[k + 1] if k < len(route) - 1 else 0

        d_base = ls.d[A, B] + ls.d[C, D] + ls.d[E, F]

        g4 = d_base - (ls.d[A, C] + ls.d[B, E] + ls.d[D, F])
        g5 = d_base - (ls.d[A, D] + ls.d[E, B] + ls.d[C, F])
        g6 = d_base - (ls.d[A, D] + ls.d[E, C] + ls.d[B, F])
        g7 = d_base - (ls.d[A, E] + ls.d[D, B] + ls.d[C, F])

        gains = [g4, g5, g6, g7]
        best_g = max(gains)
        if best_g * ls.C > 1e-4:
            case = gains.index(best_g)
            s1 = route[: i + 1]
            s2 = route[i + 1 : j + 1]
            s3 = route[j + 1 : k + 1]
            s4 = route[k + 1 :]

            if case == 0:
                route[:] = s1 + s2[::-1] + s3[::-1] + s4
            elif case == 1:
                route[:] = s1 + s3 + s2 + s4
            elif case == 2:
                route[:] = s1 + s3 + s2[::-1] + s4
            elif case == 3:
                route[:] = s1 + s3[::-1] + s2 + s4

            ls._update_map({r_u})
            return True
    return False
