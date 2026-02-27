"""
Relocate Operator Module.

This module implements the relocate operator, which moves a single node
from its current position to a new position (after another node) in the
same or a different route.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.move.relocate import move_relocate
    >>> improved = move_relocate(ls, u=1, v=2, r_u=0, p_u=1, r_v=0, p_v=5)
"""


def move_relocate(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    Relocate operator: move node u to a position after node v.

    Removes node u from route r_u and inserts it immediately after node v
    in route r_v. Can be inter-route or intra-route. Only applies the move
    if it improves total cost by a threshold margin.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Node to relocate.
        v: Node after which u will be inserted.
        r_u: Index of the route containing u.
        p_u: Position of u in route r_u.
        r_v: Index of the route containing v (target route).
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the relocation was applied (improving), False otherwise.
    """
    if r_u == r_v and (p_u == p_v + 1):
        return False
    dem_u = ls.waste.get(u, 0)

    if r_u != r_v and ls._get_load_cached(r_v) + dem_u > ls.Q:
        return False

    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]
    prev_u = route_u[p_u - 1] if p_u > 0 else 0
    next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

    delta = -ls.d[prev_u, u] - ls.d[u, next_u] + ls.d[prev_u, next_u]
    delta -= ls.d[v, v_next]
    delta += ls.d[v, u] + ls.d[u, v_next]

    if delta * ls.C < -1e-4:
        ls.routes[r_u].pop(p_u)
        if r_u == r_v and p_u < p_v:
            p_v -= 1
        ls.routes[r_v].insert(p_v + 1, u)
        ls._update_map({r_u, r_v})
        return True
    return False
