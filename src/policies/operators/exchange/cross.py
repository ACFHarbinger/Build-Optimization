"""
Cross-Exchange Operator Module.

This module implements the Cross-Exchange operator, which swaps segments of
arbitrary length between two different routes, preserving the internal order
of nodes within each segment.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.exchange.cross import cross_exchange
    >>> improved = cross_exchange(ls, r_a=0, seg_a_start=1, seg_a_len=2, ...)
"""

from typing import Any


def cross_exchange(
    ls: Any,
    r_a: int,
    seg_a_start: int,
    seg_a_len: int,
    r_b: int,
    seg_b_start: int,
    seg_b_len: int,
) -> bool:
    """
    Cross-Exchange (λ-interchange): Swap segments between two routes.

    Exchanges a segment of customers from Route A with a segment from Route B.
    The internal order of both segments is preserved.

    Args:
        ls: LocalSearch instance.
        r_a: Index of first route.
        seg_a_start: Start position of segment in Route A.
        seg_a_len: Length of segment in Route A.
        r_b: Index of second route.
        seg_b_start: Start position of segment in Route B.
        seg_b_len: Length of segment in Route B.

    Returns:
        bool: True if an improving exchange was applied.
    """
    if r_a == r_b:
        return False

    route_a = ls.routes[r_a]
    route_b = ls.routes[r_b]

    if seg_a_start + seg_a_len > len(route_a):
        return False
    if seg_b_start + seg_b_len > len(route_b):
        return False

    # Extract segments
    seg_a = route_a[seg_a_start : seg_a_start + seg_a_len]
    seg_b = route_b[seg_b_start : seg_b_start + seg_b_len]

    # Check capacity feasibility
    waste_a = sum(ls.waste.get(n, 0) for n in seg_a)
    waste_b = sum(ls.waste.get(n, 0) for n in seg_b)

    load_a = ls._calc_load_fresh(route_a)
    load_b = ls._calc_load_fresh(route_b)

    new_load_a = load_a - waste_a + waste_b
    new_load_b = load_b - waste_b + waste_a

    if new_load_a > ls.Q or new_load_b > ls.Q:
        return False

    # Calculate delta
    # Route A: remove seg_a, insert seg_b
    a_prev = route_a[seg_a_start - 1] if seg_a_start > 0 else 0
    a_next = route_a[seg_a_start + seg_a_len] if seg_a_start + seg_a_len < len(route_a) else 0

    removal_a = ls.d[a_prev, seg_a[0]] + ls.d[seg_a[-1], a_next]
    insertion_a = ls.d[a_prev, seg_b[0]] + ls.d[seg_b[-1], a_next] if seg_b else ls.d[a_prev, a_next]

    # Route B: remove seg_b, insert seg_a
    b_prev = route_b[seg_b_start - 1] if seg_b_start > 0 else 0
    b_next = route_b[seg_b_start + seg_b_len] if seg_b_start + seg_b_len < len(route_b) else 0

    removal_b = ls.d[b_prev, seg_b[0]] + ls.d[seg_b[-1], b_next]
    insertion_b = ls.d[b_prev, seg_a[0]] + ls.d[seg_a[-1], b_next] if seg_a else ls.d[b_prev, b_next]

    delta = (insertion_a - removal_a) + (insertion_b - removal_b)

    if delta * ls.C < -1e-4:
        # Apply exchange
        new_route_a = route_a[:seg_a_start] + seg_b + route_a[seg_a_start + seg_a_len :]
        new_route_b = route_b[:seg_b_start] + seg_a + route_b[seg_b_start + seg_b_len :]

        ls.routes[r_a] = new_route_a
        ls.routes[r_b] = new_route_b
        ls._update_map({r_a, r_b})
        return True

    return False
