"""
Swap Star Operator Module.

This module implements the Swap* operator (inter-route swap) for VRP.
It attempts to swap two nodes between different routes to improve the objective.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.route.swap_star import SwapStar
    >>> operator = SwapStar(...)
    >>> new_solution = operator.apply(solution)

Removes u from route r_u and v from route r_v, then reinserts each node
into the other route at the best position. Only applies the move if it
improves the total cost by a threshold margin.

Args:
    ls: LocalSearch instance containing routes and distance matrix.
    u: Node to remove from route r_u.
    v: Node to remove from route r_v.
    r_u: Index of the route containing u.
    p_u: Position of u in route r_u.
    r_v: Index of the route containing v.
    p_v: Position of v in route r_v.

Returns:
    bool: True if the swap was applied (improving), False otherwise.
"""


def move_swap_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """SWAP* inter-route operator: swap nodes u and v between different routes.

    Removes u from route r_u and v from route r_v, then reinserts each node
    into the other route at the best position. Only applies the move if it
    improves the total cost by a threshold margin.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Node to remove from route r_u.
        v: Node to remove from route r_v.
        r_u: Index of the route containing u.
        p_u: Position of u in route r_u.
        r_v: Index of the route containing v.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the swap was applied (improving), False otherwise.
    """
    waste_u = ls.waste.get(u, 0)
    waste_v = ls.waste.get(v, 0)

    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]

    prev_u = route_u[p_u - 1] if p_u > 0 else 0
    next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    gain_rem_u = ls.d[prev_u, u] + ls.d[u, next_u] - ls.d[prev_u, next_u]

    prev_v = route_v[p_v - 1] if p_v > 0 else 0
    next_v = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0
    gain_rem_v = ls.d[prev_v, v] + ls.d[v, next_v] - ls.d[prev_v, next_v]

    temp_rv = route_v[:]
    temp_rv.pop(p_v)
    if ls._calc_load_fresh(temp_rv) + waste_u > ls.Q:
        return False

    best_delta_u = float("inf")
    best_pos_u = -1

    for pos in range(len(temp_rv) + 1):
        prev = temp_rv[pos - 1] if pos > 0 else 0
        nxt = temp_rv[pos] if pos < len(temp_rv) else 0
        delta = ls.d[prev, u] + ls.d[u, nxt] - ls.d[prev, nxt]
        if delta < best_delta_u:
            best_delta_u = delta
            best_pos_u = pos

    temp_ru = route_u[:]
    temp_ru.pop(p_u)
    if ls._calc_load_fresh(temp_ru) + waste_v > ls.Q:
        return False

    best_delta_v = float("inf")
    best_pos_v = -1

    for pos in range(len(temp_ru) + 1):
        prev = temp_ru[pos - 1] if pos > 0 else 0
        nxt = temp_ru[pos] if pos < len(temp_ru) else 0
        delta = ls.d[prev, v] + ls.d[v, nxt] - ls.d[prev, nxt]
        if delta < best_delta_v:
            best_delta_v = delta
            best_pos_v = pos

    total_delta = -gain_rem_u - gain_rem_v + best_delta_u + best_delta_v

    if total_delta * ls.C < -1e-4:
        ls.routes[r_u] = temp_ru
        ls.routes[r_u].insert(best_pos_v, v)
        ls.routes[r_v] = temp_rv
        ls.routes[r_v].insert(best_pos_u, u)
        ls._update_map({r_u, r_v})
        return True
    return False
