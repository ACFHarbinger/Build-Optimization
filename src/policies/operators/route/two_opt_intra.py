"""
2-opt Intra-Route Operator Module.

This module implements the 2-opt intra-route operator, which reverses a segment
of a route to eliminate crossing edges and reduce total tour length.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.route.two_opt_intra import move_2opt_intra
    >>> improved = move_2opt_intra(ls, u, v, r_u, p_u, r_v, p_v)
"""


def move_2opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    2-opt intra-route operator: reverse a segment within a route.

    Reverses the segment between positions p_u+1 and p_v (inclusive)
    in route r_u. Only applies the move if it reduces total cost.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Start node of the segment (edge u->next_u broken).
        v: End node of the segment (edge v->next_v broken).
        r_u: Index of the route (must equal r_v for intra-route).
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.

    Returns:
        bool: True if the reversal was applied (improving), False otherwise.
    """
    if p_u >= p_v:
        return False
    if p_u + 1 == p_v:
        return False

    route = ls.routes[r_u]
    u_next = route[p_u + 1]
    v_next = route[p_v + 1] if p_v < len(route) - 1 else 0

    delta = -ls.d[u, u_next] - ls.d[v, v_next] + ls.d[u, v] + ls.d[u_next, v_next]

    if delta * ls.C < -1e-4:
        segment = route[p_u + 1 : p_v + 1]
        route[p_u + 1 : p_v + 1] = segment[::-1]
        ls._update_map({r_u})
        return True
    return False
