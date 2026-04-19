"""
k-opt Intra-Route Operator Module.

This module implements the generalized k-opt intra-route operator, which removes
k edges from a route and reconnects the segments optimally to reduce tour length.

Provides the unified ``move_kopt_intra`` function and convenience wrappers:

- ``move_2opt_intra`` → delegates to ``move_kopt_intra(k=2)``
- ``move_3opt_intra`` → delegates to ``move_kopt_intra(k=3, rng=rng)``

Higher values of *k* systematically enumerate all permutations and
orientations of the middle segments and apply the best improving move.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.intra_route.k_opt import move_kopt_intra
    >>> improved = move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=2)
    >>> improved = move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=3, rng=rng)
    >>> improved = move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=4, rng=rng)
    >>> improved_route = two_opt_route(route, dist_matrix)
"""

import itertools
from random import Random
from typing import List, Optional, Tuple

import numpy as np


def two_opt_route(
    route: List[int],
    dist_matrix: np.ndarray,
    max_iter: int = 200,
    exclude_depot: bool = False,
) -> List[int]:
    """
    Apply 2-opt to a single route (iterative, first-improvement).

    Args:
        route:         Ordered node list, depot excluded.
        dist_matrix:   Full distance matrix (index 0 = depot).
        max_iter:      Maximum improvement passes.
        exclude_depot: If True, skips edges connected to node 0.

    Returns:
        Improved route (depot excluded).
    """
    if len(route) < 2:
        return route

    route = list(route)
    full = [0] + route + [0]
    n = len(full)

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if exclude_depot and (i == 1 or j == n - 2):
                    continue
                # Edges: full[i-1]->full[i] and full[j]->full[j+1]
                delta = (
                    dist_matrix[full[i - 1], full[j]]
                    + dist_matrix[full[i], full[j + 1]]
                    - dist_matrix[full[i - 1], full[i]]
                    - dist_matrix[full[j], full[j + 1]]
                )
                if delta < -1e-9:
                    full[i : j + 1] = full[i : j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return full[1:-1]


def three_opt_route(
    route: List[int],
    dist_matrix: np.ndarray,
    max_iter: int = 50,
    exclude_depot: bool = False,
) -> List[int]:
    """
    Apply 3-opt to a single route (iterative, first-improvement).

    Args:
        route:         Ordered node list, depot excluded.
        dist_matrix:   Full distance matrix (index 0 = depot).
        max_iter:      Maximum improvement passes.
        exclude_depot: If True, skips edges connected to node 0.

    Returns:
        Improved route (depot excluded).
    """
    if len(route) < 4:
        return two_opt_route(route, dist_matrix, max_iter, exclude_depot)

    def _route_cost(r: List[int]) -> float:
        f = [0] + r + [0]
        cost = 0.0
        for i in range(len(f) - 1):
            if exclude_depot and (i == 0 or i == len(f) - 2):
                continue
            cost += dist_matrix[f[i], f[i + 1]]
        return cost

    for _ in range(max_iter):
        improved = False
        n = len(route)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    A = route[: i + 1]
                    B = route[i + 1 : j + 1]
                    C = route[j + 1 : k + 1]
                    D = route[k + 1 :]

                    curr_cost = _route_cost(route)
                    candidates = [
                        A + B[::-1] + C + D,
                        A + B + C[::-1] + D,
                        A + C + B + D,
                        A + C + B[::-1] + D,
                    ]
                    for cand in candidates:
                        if _route_cost(cand) < curr_cost - 1e-9:
                            route = cand
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return route


def move_2opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    2-opt intra-route operator: reverse a segment within a route.

    Convenience wrapper for ``move_kopt_intra(k=2)``.

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
    return move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=2)


def move_3opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: Random) -> bool:
    """
    3-opt intra-route operator: reconnect three segments within a route.

    Convenience wrapper for ``move_kopt_intra(k=3, rng=rng)``.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First cut point node.
        v: Second cut point node.
        r_u: Index of the route.
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.
        rng: Random number generator.

    Returns:
        bool: True if a 3-opt move was applied (improving), False otherwise.
    """
    return move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=3, rng=rng)


def move_kopt_intra(
    ls,
    u: int,
    v: int,
    r_u: int,
    p_u: int,
    r_v: int,
    p_v: int,
    k: int = 2,
    rng: Optional[Random] = None,
    exclude_depot: bool = False,
) -> bool:
    """
    Generalized k-opt intra-route operator.

    Removes *k* edges from a single route and reconnects the resulting
    segments in the best improving configuration found. Subsumes 2-opt
    (segment reversal), 3-opt (triple-edge reconnection), and higher-order
    moves.

    For *k* cut points the route is split into *k* + 1 segments. The head
    and tail segments stay fixed; the *k* − 1 middle segments are tried in
    every permutation and orientation. The configuration with the largest
    cost saving is applied.

    Complexity per call:
        - k=2: O(1) — single delta check.
        - k=3: O(1) — 4 reconnection patterns plus up to 5 random samples.
        - k≥4: O((k−1)! · 2^(k−1)) reconnection patterns per sample.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First breakpoint node.
        v: Second breakpoint node.
        r_u: Index of the route containing u.
        p_u: Position of u in the route.
        r_v: Index of the route containing v (must equal r_u for intra-route).
        p_v: Position of v in the route.
        k: Number of edges to remove (≥ 2). Defaults to 2.
        rng: Random number generator, required when ``k >= 3``.

    Returns:
        bool: True if an improving move was applied, False otherwise.

    Raises:
        ValueError: If *k* < 2.
        ValueError: If *k* >= 3 and *rng* is not provided.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    if k == 2:
        return _apply_2opt(ls, u, v, r_u, p_u, p_v, exclude_depot)

    if rng is None:
        raise ValueError("rng is required for k >= 3")

    if k == 3:
        return _apply_3opt(ls, u, v, r_u, p_u, p_v, rng, exclude_depot=exclude_depot)

    return _apply_kopt(ls, r_u, p_u, p_v, k, rng, exclude_depot=exclude_depot)


# ---------------------------------------------------------------------------
# Private implementations
# ---------------------------------------------------------------------------


def _apply_2opt(ls, u: int, v: int, r_u: int, p_u: int, p_v: int, exclude_depot: bool = False) -> bool:
    """
    2-opt: reverse the segment between p_u+1 and p_v (inclusive).

    Only applies the move if it reduces total cost.
    """
    if p_u >= p_v:
        return False
    if p_u + 1 == p_v:
        return False

    route = ls.routes[r_u]
    u_next = route[p_u + 1]
    if exclude_depot and p_v == len(route) - 1:
        return False
    v_next = route[p_v + 1] if p_v < len(route) - 1 else 0

    delta = -ls.d[u, u_next] - ls.d[v, v_next] + ls.d[u, v] + ls.d[u_next, v_next]

    if delta * ls.C < -1e-4:
        segment = route[p_u + 1 : p_v + 1]
        route[p_u + 1 : p_v + 1] = segment[::-1]
        ls._update_map({r_u})
        return True
    return False


def _apply_3opt(
    ls, u: int, v: int, r_u: int, p_u: int, p_v: int, rng: Random, n_attempts: int = 5, exclude_depot: bool = False
) -> bool:
    """
    3-opt: remove three edges and try all reconnection patterns.

    Randomly selects a third cut point and evaluates four reconnection
    patterns. Applies the best improving move if found.
    """
    route = ls.routes[r_u]
    if len(route) < 4:
        return False
    if p_u > p_v:
        p_u, p_v = p_v, p_u
        u, v = v, u

    for _ in range(n_attempts):
        p_w = rng.randint(0, len(route) - 1)
        if p_w in {p_u, p_v, p_u + 1, p_v + 1, p_u - 1, p_v - 1}:
            continue

        idx = sorted([p_u, p_v, p_w])
        i, j, k = idx[0], idx[1], idx[2]

        if exclude_depot and k == len(route) - 1:
            continue

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


def _apply_kopt(
    ls, r_u: int, p_u: int, p_v: int, k: int, rng: Random, n_attempts: int = 5, exclude_depot: bool = False
) -> bool:
    """
    General k-opt for k >= 4.

    Selects *k* cut points by augmenting the two given positions (``p_u``,
    ``p_v``) with ``k − 2`` randomly sampled additional points.  The route
    is sliced into a fixed head, *k* − 1 permutable middle segments, and a
    fixed tail.  Every permutation × orientation of the middle segments is
    evaluated and the best improving configuration is applied.

    Args:
        ls: LocalSearch instance.
        r_u: Route index.
        p_u: First given position.
        p_v: Second given position.
        k: Number of edges to remove (>= 4).
        rng: Random number generator.
        n_attempts: Number of random cut-point sampling attempts.
    """
    route = ls.routes[r_u]
    # A k-opt move on a route needs at least k + 1 nodes
    if len(route) < k + 1:
        return False

    i, j = (p_u, p_v) if p_u < p_v else (p_v, p_u)

    # Multiple random attempts
    for _ in range(n_attempts):
        cuts = _sample_cuts(len(route), i, j, k, rng)
        if cuts is None:
            return False

        head, middle, tail = _get_segments(route, cuts)
        if not middle:
            continue

        if exclude_depot and not tail:
            continue

        original_cost = _connection_cost(ls.d, head, middle, tail, exclude_depot)
        best_gain, best_config = _find_best_config(ls, head, middle, tail, original_cost, exclude_depot)

        # Apply the best configuration if improving
        if best_config is not None and best_gain * ls.C > 1e-4:
            _apply_config(route, head, middle, tail, best_config)
            ls._update_map({r_u})
            return True

    return False


def _sample_cuts(n: int, p_u: int, p_v: int, k: int, rng: Random) -> Optional[List[int]]:
    """Sample k-2 additional cut points around p_u and p_v."""
    extra_needed = k - 2
    forbidden = {p_u, p_v}
    for p in (p_u, p_v):
        if p > 0:
            forbidden.add(p - 1)
        if p < n - 1:
            forbidden.add(p + 1)

    available = [p for p in range(n) if p not in forbidden]
    if len(available) < extra_needed:
        return None

    extra = rng.sample(available, extra_needed)
    return sorted([p_u, p_v] + extra)


def _get_segments(route: List[int], cuts: List[int]) -> Tuple[List[int], List[List[int]], List[int]]:
    """Slice the route into head, middle segments, and tail."""
    segments: List[List[int]] = []
    segments.append(route[: cuts[0] + 1])
    for ci in range(len(cuts) - 1):
        segments.append(route[cuts[ci] + 1 : cuts[ci + 1] + 1])
    segments.append(route[cuts[-1] + 1 :])
    return segments[0], segments[1:-1], segments[-1]


def _find_best_config(
    ls, head: List[int], middle: List[List[int]], tail: List[int], original_cost: float, exclude_depot: bool = False
) -> Tuple[float, Optional[Tuple[Tuple[int, ...], Tuple[bool, ...]]]]:
    """Enumerate all permutations and orientations to find best gain."""
    best_gain = 0.0
    best_config = None
    n_middle = len(middle)
    identity_perm = tuple(range(n_middle))
    for perm in itertools.permutations(range(n_middle)):
        for orient_bits in range(1 << n_middle):
            reversals = tuple(bool(orient_bits & (1 << b)) for b in range(n_middle))
            if perm == identity_perm and not any(reversals):
                continue

            reordered = []
            for idx, seg_idx in enumerate(perm):
                seg = middle[seg_idx]
                reordered.append(seg[::-1] if reversals[idx] else seg)

            new_cost = _connection_cost(ls.d, head, reordered, tail, exclude_depot)
            gain = original_cost - new_cost

            if gain > best_gain:
                best_gain = gain
                best_config = (perm, reversals)
    return best_gain, best_config


def _apply_config(
    route: List[int],
    head: List[int],
    middle: List[List[int]],
    tail: List[int],
    config: Tuple[Tuple[int, ...], Tuple[bool, ...]],
) -> None:
    """Apply the selected configuration to the route."""
    perm, reversals = config
    new_route: List[int] = list(head)
    for idx, seg_idx in enumerate(perm):
        seg = middle[seg_idx]
        if reversals[idx]:
            new_route.extend(seg[::-1])
        else:
            new_route.extend(seg)
    new_route.extend(tail)
    route[:] = new_route


def _connection_cost(
    d, head: List[int], middle: List[List[int]], tail: List[int], exclude_depot: bool = False
) -> float:
    """
    Compute the total cost of edges connecting consecutive segments.

    Only counts the *k* edges that a k-opt move breaks/reconnects,
    not internal segment costs (which stay constant).

    Args:
        d: Distance matrix (2-D array-like).
        head: Head segment (fixed).
        middle: List of middle segments in their current order.
        tail: Tail segment (fixed).

    Returns:
        Sum of inter-segment edge costs.
    """
    cost = 0.0
    all_segs = [head] + middle + [tail]

    for i in range(len(all_segs) - 1):
        seg_a = all_segs[i]
        seg_b = all_segs[i + 1]
        # Last node of seg_a -> first node of seg_b
        if exclude_depot and (not seg_a or not seg_b):
            continue

        node_a = seg_a[-1] if seg_a else 0
        node_b = seg_b[0] if seg_b else 0
        cost += d[node_a, node_b]

    return cost
