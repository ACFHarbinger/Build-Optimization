"""
2-opt* Inter-Route Operator Module.

This module implements the 2-opt* operator, which exchanges the tails of two
different routes. It is a powerful operator for reducing the number of routes
and balancing loads.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.route.two_opt_star import move_2opt_star
    >>> improved = move_2opt_star(ls, u, v, r_u, p_u, r_v, p_v)
"""


def move_2opt_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    2-opt* inter-route operator: exchange tails between two routes.

    Cuts route r_u after node u and route r_v after node v, then swaps
    the tails. New routes become (start_u + tail_v) and (start_v + tail_u).

    Args:
        ls: LocalSearch instance.
        u: Cut point in route r_u.
        v: Cut point in route r_v.
        r_u: Index of the first route.
        p_u: Position of u in route r_u.
        r_v: Index of the second route.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the exchange was applied (improving), False otherwise.
    """
    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]

    tail_u = route_u[p_u + 1 :]
    tail_v = route_v[p_v + 1 :]

    l_head_u = ls._calc_load_fresh(route_u[: p_u + 1])
    l_head_v = ls._calc_load_fresh(route_v[: p_v + 1])
    l_tail_u = ls._get_load_cached(r_u) - l_head_u
    l_tail_v = ls._get_load_cached(r_v) - l_head_v

    if l_head_u + l_tail_v > ls.Q or l_head_v + l_tail_u > ls.Q:
        return False

    u_next = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

    delta = -ls.d[u, u_next] - ls.d[v, v_next] + ls.d[u, v_next] + ls.d[v, u_next]

    if delta * ls.C < -1e-4:
        new_ru = route_u[: p_u + 1] + tail_v
        new_rv = route_v[: p_v + 1] + tail_u
        ls.routes[r_u] = new_ru
        ls.routes[r_v] = new_rv
        ls._update_map({r_u, r_v})
        return True
    return False
