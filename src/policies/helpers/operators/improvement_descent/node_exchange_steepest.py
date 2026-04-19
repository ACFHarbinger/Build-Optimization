"""
Steepest-Descent Node-Exchange (Swap) Intensification Operator.

Evaluates every pairwise swap of customer nodes — both within a route
(intra-route) and across different routes (inter-route) — and applies the
globally best improving exchange at each iteration, repeating until a strict
swap local minimum is reached.

Swapping two non-adjacent nodes u and v changes exactly four boundary edges:

    Original:  …prev_u → u → next_u… and …prev_v → v → next_v…
    After:     …prev_u → v → next_u… and …prev_v → u → next_v…

    Δ = d[prev_u,v] + d[v,next_u] + d[prev_v,u] + d[u,next_v]
      − d[prev_u,u] − d[u,next_u] − d[prev_v,v] − d[v,next_v]

Adjacent same-route nodes (|p_u − p_v| = 1) are skipped here because their
reversal is exactly the 2-opt move of a 2-node segment — already covered by
:func:`~intensification.steepest_two_opt.two_opt_steepest`.

Inter-route swaps are subject to capacity feasibility: both receiving routes
must still satisfy the load limit after the exchange.

Because every swap preserves the set of visited customers, revenue is unchanged.
The CVRP and VRPP stopping conditions are therefore equivalent, with the delta
scaled by C for unit consistency.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import node_exchange_steepest
    >>> improved = node_exchange_steepest(routes, dist_matrix, wastes, capacity)
    >>> improved = node_exchange_steepest_profit(routes, dist_matrix, wastes,
    ...                                          capacity, R=1.0, C=0.5)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Internal search engine
# ---------------------------------------------------------------------------


def _find_best_swap(  # noqa: C901
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    C: float,
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """Find the globally best improving pairwise node swap.

    Considers every ordered pair (u at r_u/p_u, v at r_v/p_v) with either
    ``r_u < r_v`` (all inter-route pairs) or ``r_u == r_v`` and ``p_u < p_v``
    with ``|p_u − p_v| > 1`` (non-adjacent intra-route pairs).

    For inter-route swaps both affected loads are checked against *capacity*.

    Args:
        routes: Current plan.
        loads: Per-route cumulative load (parallel to *routes*).
        dist_matrix: Distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        C: Cost coefficient.

    Returns:
        ``(best_delta, (r_u, p_u, r_v, p_v))`` or ``(0.0, None)``.
    """
    best_delta = -1e-6
    best_move: Optional[Tuple[int, int, int, int]] = None
    d = dist_matrix
    n_routes = len(routes)

    for r_u in range(n_routes):
        route_u = routes[r_u]
        n_u = len(route_u)

        for p_u in range(n_u):
            u = route_u[p_u]
            prev_u = route_u[p_u - 1] if p_u > 0 else 0
            next_u = route_u[p_u + 1] if p_u + 1 < n_u else 0
            waste_u = wastes.get(u, 0.0)

            # Iterate r_v >= r_u to avoid evaluating the same pair twice
            for r_v in range(r_u, n_routes):
                route_v = routes[r_v]
                n_v = len(route_v)

                # For same route start after p_u; for different routes start at 0
                p_v_start = p_u + 1 if r_u == r_v else 0

                for p_v in range(p_v_start, n_v):
                    # Skip adjacent same-route nodes (handled by 2-opt)
                    if r_u == r_v and p_v - p_u == 1:
                        continue

                    v = route_v[p_v]
                    prev_v = route_v[p_v - 1] if p_v > 0 else 0
                    next_v = route_v[p_v + 1] if p_v + 1 < n_v else 0
                    waste_v = wastes.get(v, 0.0)

                    # Capacity feasibility for inter-route swaps only
                    if r_u != r_v:
                        if loads[r_u] - waste_u + waste_v > capacity:
                            continue
                        if loads[r_v] - waste_v + waste_u > capacity:
                            continue

                    # Edge-cost delta when u and v exchange positions.
                    # Valid for non-adjacent pairs (same or different routes).
                    delta = (
                        d[prev_u, v]
                        + d[v, next_u]
                        + d[prev_v, u]
                        + d[u, next_v]
                        - d[prev_u, u]
                        - d[u, next_u]
                        - d[prev_v, v]
                        - d[v, next_v]
                    ) * C

                    if delta < best_delta:
                        best_delta = delta
                        best_move = (r_u, p_u, r_v, p_v)

    return best_delta, best_move


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def node_exchange_steepest(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    C: float = 1.0,
    max_iter: int = 300,
) -> List[List[int]]:
    """Drive the solution to a strict node-swap local minimum.

    At every iteration the globally best pairwise node exchange is identified
    and applied.  The search terminates once no single swap reduces total
    travel cost by more than 1e-6.

    Adjacent same-route pairs are excluded (they are equivalent to 2-opt moves
    and should be handled by :func:`two_opt_steepest`).  Inter-route swaps are
    subject to capacity feasibility: if exchanging demands would violate either
    route's capacity limit, the move is skipped.

    Swapping nodes never changes the set of visited customers, so revenue and
    route loads are preserved by construction.  For VRPP problems pass ``C``
    as the cost-per-km to keep comparisons in profit units.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix ``(N+1, N+1)``, depot at index 0.
        wastes: Mapping from node index to demand.
        capacity: Maximum vehicle load.
        C: Cost per unit of distance.  Use ``1.0`` for CVRP; the actual cost
            coefficient for VRPP profit weighting.
        max_iter: Safety upper bound on iterations.

    Returns:
        List[List[int]]: Routes at a node-swap local minimum.  The input
            *routes* is never mutated.

    Example:
        >>> improved = node_exchange_steepest(routes, dist_matrix, wastes, capacity)
    """
    working = [list(r) for r in routes if r]
    if not working:
        return working

    loads = [sum(wastes.get(n, 0.0) for n in r) for r in working]

    for _ in range(max_iter):
        _, move = _find_best_swap(working, loads, dist_matrix, wastes, capacity, C)
        if move is None:
            break

        r_u, p_u, r_v, p_v = move
        u = working[r_u][p_u]
        v = working[r_v][p_v]
        waste_u = wastes.get(u, 0.0)
        waste_v = wastes.get(v, 0.0)

        # Apply swap in-place
        working[r_u][p_u] = v
        working[r_v][p_v] = u

        # Update loads only for inter-route swaps
        if r_u != r_v:
            loads[r_u] = loads[r_u] - waste_u + waste_v
            loads[r_v] = loads[r_v] - waste_v + waste_u

    return [r for r in working if r]


def node_exchange_steepest_profit(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    max_iter: int = 300,
) -> List[List[int]]:
    """Steepest-descent node swap for VRPP profit maximisation.

    Delegates to :func:`node_exchange_steepest` with ``C`` forwarded.  Revenue
    is unchanged by any swap (same customers visited, same waste collected).

    Args:
        routes: Current plan.
        dist_matrix: Distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Vehicle capacity.
        R: Revenue per unit waste (unused; revenue unchanged by swaps).
        C: Cost per unit distance.
        max_iter: Maximum iterations.

    Returns:
        List[List[int]]: Routes at a node-swap local minimum.
    """
    return node_exchange_steepest(routes, dist_matrix, wastes, capacity, C=C, max_iter=max_iter)
