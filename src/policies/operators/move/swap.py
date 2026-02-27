"""
Swap Operator Module.

This module implements the swap operator, which exchanges the positions of
two nodes, either within the same route or between two different routes.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.move.swap import move_swap
    >>> improved = move_swap(ls, u=1, v=2, r_u=0, p_u=1, r_v=1, p_v=3)
"""


def move_swap(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    Swap operator: exchange positions of nodes u and v.

    Swaps node u (in route r_u) with node v (in route r_v). Can be
    inter-route or intra-route swap. Only applies the move if it
    improves total cost by a threshold margin.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First node to swap.
        v: Second node to swap.
        r_u: Index of the route containing u.
        p_u: Position of u in route r_u.
        r_v: Index of the route containing v.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the swap was applied (improving), False otherwise.
    """
    if r_u == r_v and abs(p_u - p_v) <= 1:
        return False

    dem_u = ls.waste.get(u, 0)
    dem_v = ls.waste.get(v, 0)

    if r_u != r_v:
        if ls._get_load_cached(r_u) - dem_u + dem_v > ls.Q:
            return False
        if ls._get_load_cached(r_v) - dem_v + dem_u > ls.Q:
            return False

    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]

    prev_u = route_u[p_u - 1] if p_u > 0 else 0
    next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    prev_v = route_v[p_v - 1] if p_v > 0 else 0
    next_v = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

    delta = -ls.d[prev_u, u] - ls.d[u, next_u] - ls.d[prev_v, v] - ls.d[v, next_v]
    delta += ls.d[prev_u, v] + ls.d[v, next_u] + ls.d[prev_v, u] + ls.d[u, next_v]

    if delta * ls.C < -1e-4:
        ls.routes[r_u][p_u] = v
        ls.routes[r_v][p_v] = u
        ls._update_map({r_u, r_v})
        return True
    return False
