"""
Cross-Exchange Operator, Improved CROSS-Exchange (I-CROSS) Operator,
and Lambda-Interchange Operator Module.

This module implements:

- The Cross-Exchange operator, which swaps segments of arbitrary
    length between two different routes, preserving the internal
    order of nodes within each segment.

- The Lambda-Interchange neighborhood search, which systematically
    explores cross-exchange moves between all pairs of routes with
    segment lengths up to a specified maximum (lambda_max).

- The I-CROSS operator, which extends the standard CROSS-exchange
    by also evaluating *inverted* (reversed) segments before swapping.
    For each candidate pair of segments, four configurations are tried:
        1. Standard swap (neither segment reversed)
        2. Segment A reversed, Segment B as-is
        3. Segment A as-is, Segment B reversed
        4. Both segments reversed
    The configuration with the best improving delta is then applied.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.inter_route.cross_exchange import cross_exchange
    >>> improved = cross_exchange(ls, r_a=0, seg_a_start=1, seg_a_len=2, ...)
    >>> from logic.src.policies.helpers.operators.inter_route.cross_exchange import lambda_interchange
    >>> improved = lambda_interchange(ls, lambda_max=2)
    >>> from logic.src.policies.helpers.operators.inter_route.cross_exchange import improved_cross_exchange
    >>> improved = improved_cross_exchange(ls, r_a=0, seg_a_start=1, seg_a_len=2,
    ...                                    r_b=1, seg_b_start=0, seg_b_len=3)
"""

from typing import Any, List, Optional, Tuple


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

    removal_a = ls.d[a_prev, seg_a[0]] + ls.d[seg_a[-1], a_next] if seg_a else ls.d[a_prev, a_next]
    insertion_a = ls.d[a_prev, seg_b[0]] + ls.d[seg_b[-1], a_next] if seg_b else ls.d[a_prev, a_next]

    # Route B: remove seg_b, insert seg_a
    b_prev = route_b[seg_b_start - 1] if seg_b_start > 0 else 0
    b_next = route_b[seg_b_start + seg_b_len] if seg_b_start + seg_b_len < len(route_b) else 0

    removal_b = ls.d[b_prev, seg_b[0]] + ls.d[seg_b[-1], b_next] if seg_b else ls.d[b_prev, b_next]
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


def lambda_interchange(
    ls: Any,
    lambda_max: int = 2,
) -> bool:
    """
    λ-Interchange neighborhood search.

    Systematically explores cross-exchange moves with segments up to
    length λ. This is a wrapper that explores the full neighborhood.

    Args:
        ls: LocalSearch instance.
        lambda_max: Maximum segment length to exchange.

    Returns:
        bool: True if any improving move was found.
    """
    improved = False

    for r_a in range(len(ls.routes)):
        for r_b in range(r_a + 1, len(ls.routes)):
            route_a = ls.routes[r_a]
            route_b = ls.routes[r_b]

            for seg_a_len in range(lambda_max + 1):
                for seg_b_len in range(lambda_max + 1):
                    if seg_a_len == 0 and seg_b_len == 0:
                        continue

                    for seg_a_start in range(max(1, len(route_a) - seg_a_len + 1)):
                        for seg_b_start in range(max(1, len(route_b) - seg_b_len + 1)):
                            if cross_exchange(
                                ls,
                                r_a,
                                seg_a_start,
                                seg_a_len,
                                r_b,
                                seg_b_start,
                                seg_b_len,
                            ):
                                improved = True
                                # Restart search after improvement
                                return True

    return improved


def _seg_boundary_cost(d, prev_node: int, seg: List[int], next_node: int) -> float:
    """Cost of edges connecting prev_node → seg → next_node."""
    if not seg:
        return d[prev_node, next_node]
    return d[prev_node, seg[0]] + d[seg[-1], next_node]


def improved_cross_exchange(
    ls: Any,
    r_a: int,
    seg_a_start: int,
    seg_a_len: int,
    r_b: int,
    seg_b_start: int,
    seg_b_len: int,
) -> bool:
    """
    Improved CROSS-exchange: swap segments with optional inversion.

    Evaluates standard and inverted segment swaps between two routes,
    applying the best improving configuration.

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

    seg_a = route_a[seg_a_start : seg_a_start + seg_a_len]
    seg_b = route_b[seg_b_start : seg_b_start + seg_b_len]

    if not seg_a and not seg_b:
        return False

    # Capacity feasibility
    waste_a = sum(ls.waste.get(n, 0) for n in seg_a)
    waste_b = sum(ls.waste.get(n, 0) for n in seg_b)
    load_a = ls._calc_load_fresh(route_a)
    load_b = ls._calc_load_fresh(route_b)

    new_load_a = load_a - waste_a + waste_b
    new_load_b = load_b - waste_b + waste_a

    if new_load_a > ls.Q or new_load_b > ls.Q:
        return False

    # Neighbor nodes for delta calculation
    a_prev = route_a[seg_a_start - 1] if seg_a_start > 0 else 0
    a_next = route_a[seg_a_start + seg_a_len] if seg_a_start + seg_a_len < len(route_a) else 0
    b_prev = route_b[seg_b_start - 1] if seg_b_start > 0 else 0
    b_next = route_b[seg_b_start + seg_b_len] if seg_b_start + seg_b_len < len(route_b) else 0

    # Removal costs (constant across configurations)
    removal_a = _seg_boundary_cost(ls.d, a_prev, seg_a, a_next)
    removal_b = _seg_boundary_cost(ls.d, b_prev, seg_b, b_next)
    base_removal = removal_a + removal_b

    # Evaluate all 4 configurations: (reverse_a, reverse_b)
    configs: List[Tuple[bool, bool]] = [(False, False), (True, False), (False, True), (True, True)]
    best_delta: Optional[float] = None
    best_cfg: Optional[Tuple[bool, bool]] = None

    for rev_a, rev_b in configs:
        ins_b_into_a = seg_b[::-1] if rev_b else seg_b
        ins_a_into_b = seg_a[::-1] if rev_a else seg_a

        insert_a = _seg_boundary_cost(ls.d, a_prev, ins_b_into_a, a_next)
        insert_b = _seg_boundary_cost(ls.d, b_prev, ins_a_into_b, b_next)
        delta = (insert_a + insert_b) - base_removal

        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_cfg = (rev_a, rev_b)

    if best_delta is not None and best_cfg is not None and best_delta * ls.C < -1e-4:
        rev_a, rev_b = best_cfg
        new_seg_b = seg_b[::-1] if rev_b else seg_b
        new_seg_a = seg_a[::-1] if rev_a else seg_a

        new_route_a = route_a[:seg_a_start] + new_seg_b + route_a[seg_a_start + seg_a_len :]
        new_route_b = route_b[:seg_b_start] + new_seg_a + route_b[seg_b_start + seg_b_len :]

        ls.routes[r_a] = new_route_a
        ls.routes[r_b] = new_route_b
        ls._update_map({r_a, r_b})
        return True

    return False
