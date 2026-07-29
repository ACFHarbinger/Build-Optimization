"""
k-opt* Inter-Route Operator Module.

This module implements the generalized k-opt* inter-route operator, which
exchanges tails across *k* distinct routes by cutting each route at a given
node and reassigning the resulting tails to different routes.

Provides the unified ``move_kopt_star`` function and convenience wrappers:

- ``move_2opt_star`` → delegates to ``move_kopt_star`` with 2 cuts
- ``move_3opt_star`` → delegates to ``move_kopt_star`` with 3 cuts

Higher values of *k* enumerate all k!−1 non-identity permutations of
tail assignments and apply the best improving configuration.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.inter_route.k_opt_star import move_kopt_star
    >>> improved = move_kopt_star(ls, [(u, r_u, p_u), (v, r_v, p_v)])
    >>> improved = move_kopt_star(ls, [(u, r_u, p_u), (v, r_v, p_v), (w, r_w, p_w)])
"""

import itertools
from typing import List, Optional, Tuple

# Type alias: (node_id, route_index, position_in_route)
CutPoint = Tuple[int, int, int]


def move_2opt_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    2-opt* inter-route operator: exchange tails between two routes.

    Convenience wrapper for ``move_kopt_star`` with 2 cut points.

    Args:
        ls: LocalSearch instance.
        u: Cut point node in route r_u.
        v: Cut point node in route r_v.
        r_u: Index of the first route.
        p_u: Position of u in route r_u.
        r_v: Index of the second route.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the exchange was applied (improving), False otherwise.
    """
    return move_kopt_star(ls, [(u, r_u, p_u), (v, r_v, p_v)])


def move_3opt_star(
    ls,
    u: int,
    v: int,
    w: int,
    r_u: int,
    p_u: int,
    r_v: int,
    p_v: int,
    r_w: int,
    p_w: int,
) -> bool:
    """
    3-opt* inter-route operator: exchange tails among three routes.

    Convenience wrapper for ``move_kopt_star`` with 3 cut points.

    Args:
        ls: LocalSearch instance.
        u: Cut point node in route r_u.
        v: Cut point node in route r_v.
        w: Cut point node in route r_w.
        r_u: Index of the first route.
        p_u: Position of u in route r_u.
        r_v: Index of the second route.
        p_v: Position of v in route r_v.
        r_w: Index of the third route.
        p_w: Position of w in route r_w.

    Returns:
        bool: True if the exchange was applied (improving), False otherwise.
    """
    return move_kopt_star(ls, [(u, r_u, p_u), (v, r_v, p_v), (w, r_w, p_w)])


def move_kopt_star(ls, cuts: List[CutPoint]) -> bool:
    """
    Generalized k-opt* inter-route operator.

    Cuts *k* distinct routes at the specified nodes and tries every
    non-identity permutation of the resulting tails.  The best improving
    permutation that satisfies capacity constraints is applied.

    Complexity per call: O(k!) tail permutations.

    Args:
        ls: LocalSearch instance containing routes, distance matrix ``d``,
            capacity ``Q``, cost multiplier ``C``, and helper methods
            ``_calc_load_fresh`` / ``_get_load_cached`` / ``_update_map``.
        cuts: List of ``(node, route_index, position)`` tuples, one per
              route involved.  All route indices must be distinct.

    Returns:
        bool: True if an improving tail exchange was applied, False otherwise.

    Raises:
        ValueError: If fewer than 2 cuts are provided.
        ValueError: If any two cuts reference the same route.
    """
    k = len(cuts)
    if k < 2:
        raise ValueError(f"k-opt* requires at least 2 cuts, got {k}")

    route_indices = [c[1] for c in cuts]
    if len(set(route_indices)) != k:
        raise ValueError("All route indices must be distinct for k-opt*")

    # Extract heads, tails, and costs
    heads, tails, head_loads, tail_loads, original_cost = _extract_route_parts(ls, cuts)

    # Find the best non-identity permutation of tail assignments
    best_gain, best_perm = _find_best_tail_permutation(ls, cuts, heads, tails, head_loads, tail_loads, original_cost)

    # Apply the best permutation if improving
    if best_perm is not None and best_gain * ls.C > 1e-4:
        _apply_tail_permutation(ls, cuts, heads, tails, best_perm)
        return True

    return False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_route_parts(
    ls, cuts: List[CutPoint]
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float], float]:
    """
    Split each route at its cut point into head and tail.

    Returns:
        Tuple of (heads, tails, head_loads, tail_loads, original_connection_cost).
    """
    heads: List[List[int]] = []
    tails: List[List[int]] = []
    head_loads: List[float] = []
    tail_loads: List[float] = []
    original_cost = 0.0

    for node, r_idx, pos in cuts:
        route = ls.routes[r_idx]
        head = route[: pos + 1]
        tail = route[pos + 1 :]

        l_head = ls._calc_load_fresh(head)
        l_tail = ls._get_load_cached(r_idx) - l_head

        heads.append(head)
        tails.append(tail)
        head_loads.append(l_head)
        tail_loads.append(l_tail)

        # Cost of the edge from the cut node to the next node (or depot)
        next_node = tail[0] if tail else 0
        original_cost += ls.d[node, next_node]

    return heads, tails, head_loads, tail_loads, original_cost


def _find_best_tail_permutation(
    ls,
    cuts: List[CutPoint],
    heads: List[List[int]],
    tails: List[List[int]],
    head_loads: List[float],
    tail_loads: List[float],
    original_cost: float,
) -> Tuple[float, Optional[Tuple[int, ...]]]:
    """
    Enumerate non-identity permutations of tails and return the best.

    Returns:
        Tuple of (best_gain, best_permutation) or (0.0, None) if none improves.
    """
    k = len(cuts)
    identity = tuple(range(k))
    best_gain = 0.0
    best_perm: Optional[Tuple[int, ...]] = None

    for perm in itertools.permutations(range(k)):
        if perm == identity:
            continue

        # Check capacity: head_i gets tail_perm[i]
        feasible = True
        for i in range(k):
            if head_loads[i] + tail_loads[perm[i]] > ls.Q:
                feasible = False
                break
        if not feasible:
            continue

        # Compute new connection cost
        new_cost = 0.0
        for i in range(k):
            cut_node = cuts[i][0]
            new_tail = tails[perm[i]]
            next_node = new_tail[0] if new_tail else 0
            new_cost += ls.d[cut_node, next_node]

        gain = original_cost - new_cost
        if gain > best_gain:
            best_gain = gain
            best_perm = perm

    return best_gain, best_perm


def _apply_tail_permutation(
    ls,
    cuts: List[CutPoint],
    heads: List[List[int]],
    tails: List[List[int]],
    perm: Tuple[int, ...],
) -> None:
    """Reassemble routes using the permuted tail assignment and update state."""
    affected = set()
    for i, (_, r_idx, _) in enumerate(cuts):
        ls.routes[r_idx] = heads[i] + tails[perm[i]]
        affected.add(r_idx)
    ls._update_map(affected)
