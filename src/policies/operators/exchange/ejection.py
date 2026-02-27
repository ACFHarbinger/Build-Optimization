"""
Ejection Chain Operator Module.

This module implements the Ejection Chain operator, which is primarily used for
fleet minimization by attempting to empty a route and eject its customers into
other routes, potentially triggering a chain of displacements.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.exchange.ejection import ejection_chain
    >>> success = ejection_chain(ls, source_route=0, max_depth=5)
"""

from typing import Any, List, Optional, Tuple


def ejection_chain(
    ls: Any,
    source_route: int,
    max_depth: int = 5,
) -> bool:
    """
    Ejection Chain: Compound displacement for fleet reduction.

    Attempts to empty a route by ejecting its customers into other routes.
    When a target route is full, it triggers a chain of ejections.

    This operator is primarily used for fleet size minimization.

    Args:
        ls: LocalSearch instance.
        source_route: Index of route to empty.
        max_depth: Maximum chain depth before giving up.

    Returns:
        bool: True if the source route was successfully emptied.
    """
    if source_route >= len(ls.routes) or not ls.routes[source_route]:
        return False

    # Try to empty the source route
    route = ls.routes[source_route]
    nodes_to_eject = route[:]

    # Track successful insertions
    ejection_log: List[Tuple[int, int, int]] = []  # (node, target_route, position)

    for node in nodes_to_eject:
        inserted = _try_insert_with_chain(ls, node, source_route, max_depth, ejection_log)
        if not inserted:
            # Rollback all insertions
            _rollback_ejections(ls, ejection_log, source_route)
            return False

    # All nodes successfully ejected
    if ls.routes[source_route]:
        ls.routes[source_route] = []
    ls._update_map(set(range(len(ls.routes))))
    return True


def _try_insert_with_chain(
    ls: Any,
    node: int,
    excluded_route: int,
    depth: int,
    log: List[Tuple[int, int, int]],
) -> bool:
    """Try to insert a node, potentially triggering chain ejections."""
    if depth <= 0:
        return False

    node_waste = ls.waste.get(node, 0)

    # Find best insertion
    best_cost = float("inf")
    best_insertion: Optional[Tuple[int, int]] = None

    for r_idx, route in enumerate(ls.routes):
        if r_idx == excluded_route:
            continue

        load = ls._calc_load_fresh(route)
        if load + node_waste > ls.Q:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost = ls.d[prev, node] + ls.d[node, nxt] - ls.d[prev, nxt]

            if cost < best_cost:
                best_cost = cost
                best_insertion = (r_idx, pos)

    if best_insertion is not None:
        r, p = best_insertion
        ls.routes[r].insert(p, node)
        log.append((node, r, p))
        return True

    # No direct insertion possible - try ejection chain
    for r_idx, route in enumerate(ls.routes):
        if r_idx == excluded_route or not route:
            continue

        # Try ejecting a node from this route
        for eject_pos, eject_node in enumerate(route):
            eject_waste = ls.waste.get(eject_node, 0)
            load = ls._calc_load_fresh(route)

            # Check if we can fit the new node after ejection
            if load - eject_waste + node_waste > ls.Q:
                continue

            # Eject and try to reinsert recursively
            route.pop(eject_pos)

            # Insert new node
            best_pos = 0
            best_ins_cost = float("inf")
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost = ls.d[prev, node] + ls.d[node, nxt] - ls.d[prev, nxt]
                if cost < best_ins_cost:
                    best_ins_cost = cost
                    best_pos = pos

            route.insert(best_pos, node)
            log.append((node, r_idx, best_pos))

            # Now try to insert ejected node
            if _try_insert_with_chain(ls, eject_node, excluded_route, depth - 1, log):
                return True

            # Rollback
            route.pop(best_pos)
            route.insert(eject_pos, eject_node)
            log.pop()

    return False


def _rollback_ejections(
    ls: Any,
    log: List[Tuple[int, int, int]],
    source_route: int,
) -> None:
    """Rollback ejection chain insertions."""
    for node, r_idx, pos in reversed(log):
        if r_idx < len(ls.routes) and pos < len(ls.routes[r_idx]) and ls.routes[r_idx][pos] == node:
            ls.routes[r_idx].pop(pos)
            # Put back in source route
            ls.routes[source_route].append(node)
    log.clear()
