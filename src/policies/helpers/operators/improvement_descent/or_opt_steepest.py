"""
Steepest-Descent Or-opt Intensification Operator.

Or-opt relocates contiguous chains of 1, 2, or 3 nodes to better positions
within or across routes.  The steepest-descent variant evaluates every feasible
relocation at every iteration and applies the globally best improving move,
repeating until a strict Or-opt local minimum is reached.

The chain is always re-inserted intact (in its original node order), so no
capacity violation can arise within the chain itself.  Capacity is only checked
when the chain crosses a route boundary (inter-route relocation).

Because Or-opt repositions customers but never drops them, the set of visited
nodes does not change.  Revenue is therefore unchanged, making the CVRP and
VRPP stopping conditions equivalent: improving moves reduce distance cost by
more than an epsilon threshold (scaled by C for unit consistency).

The complexity per iteration is O(m² · n²) where m is the number of routes
and n is the average route length.  The algorithm converges quickly in practice.

Reference:
    Or, I. (1976). "Traveling Salesman-Type Combinatorial Problems and Their
    Relation to the Logistics of Regional Blood Banking." Ph.D. thesis,
    Northwestern University.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import or_opt_steepest
    >>> improved = or_opt_steepest(routes, dist_matrix, wastes, capacity)
    >>> improved = or_opt_steepest_profit(routes, dist_matrix, wastes, capacity, R=1.0, C=0.5)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_best_or_opt(  # noqa: C901
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    C: float,
    chain_lengths: Tuple[int, ...],
) -> Tuple[float, Optional[Tuple[int, int, int, int, int]]]:
    """Find the globally best Or-opt relocation across all routes and chain sizes.

    For each chain (route, length, start-position) and each feasible insertion
    slot (target route, position-within-post-removal-target), the delta is:

        delta = insertion_cost − removal_saving

    where::

        removal_saving = d[prev_c, chain[0]] + d[chain[-1], next_c] − d[prev_c, next_c]
        insertion_cost = d[v_before, chain[0]] + d[chain[-1], v_after] − d[v_before, v_after]

    Only intra-chain edge costs (which cancel) are omitted from the delta.

    Args:
        routes: Current plan.
        loads: Per-route cumulative load (parallel to *routes*).
        dist_matrix: Distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        C: Cost coefficient.  Scales delta for unit consistency.
        chain_lengths: Tuple of chain sizes to evaluate simultaneously
            (e.g. ``(1, 2, 3)`` for the full Or-opt-3 neighbourhood).

    Returns:
        ``(best_delta, (r1_idx, k, p, r2_idx, q))`` or ``(0.0, None)`` if no
        improving move exists.  The stored *q* is the insertion index in the
        **effective** target route — the post-removal version of *routes[r1_idx]*
        when ``r1_idx == r2_idx``, and *routes[r2_idx]* otherwise.  Inserting
        *after* position ``q`` (``q = -1`` means before all existing nodes)
        applies the move.
    """
    best_delta = -1e-6
    best_move: Optional[Tuple[int, int, int, int, int]] = None
    d = dist_matrix
    n_routes = len(routes)

    for r1_idx in range(n_routes):
        route1 = routes[r1_idx]
        n1 = len(route1)

        for k in chain_lengths:
            if k > n1:
                continue

            for p in range(n1 - k + 1):
                chain = route1[p : p + k]
                chain_load = sum(wastes.get(node, 0.0) for node in chain)

                prev_c = route1[p - 1] if p > 0 else 0
                next_c = route1[p + k] if p + k < n1 else 0

                # Distance saved at the removal site (without intra-chain edges)
                removal_saving = d[prev_c, chain[0]] + d[chain[-1], next_c] - d[prev_c, next_c]

                # Effective post-removal route for the intra-route case
                route_eff_intra = route1[:p] + route1[p + k :]
                n_eff_intra = len(route_eff_intra)

                for r2_idx in range(n_routes):
                    if r1_idx == r2_idx:
                        route_eff = route_eff_intra
                        n_eff = n_eff_intra
                        cap_ok = True  # Load within the same route is unchanged
                    else:
                        route_eff = routes[r2_idx]
                        n_eff = len(route_eff)
                        cap_ok = loads[r2_idx] + chain_load <= capacity

                    if not cap_ok:
                        continue

                    # Evaluate every insertion slot in the effective target route.
                    # q = -1 → insert before route_eff[0] (i.e. at position 0).
                    for q in range(-1, n_eff):
                        v_before = route_eff[q] if q >= 0 else 0
                        v_after = route_eff[q + 1] if q + 1 < n_eff else 0

                        insertion_cost = d[v_before, chain[0]] + d[chain[-1], v_after] - d[v_before, v_after]

                        delta = (insertion_cost - removal_saving) * C

                        if delta < best_delta:
                            best_delta = delta
                            best_move = (r1_idx, k, p, r2_idx, q)

    return best_delta, best_move


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def or_opt_steepest(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    C: float = 1.0,
    chain_lengths: Tuple[int, ...] = (1, 2, 3),
    max_iter: int = 500,
) -> List[List[int]]:
    """Drive the solution to a strict Or-opt local minimum.

    At every iteration the globally best chain relocation (intra or inter-route)
    is identified and applied.  Chains of lengths 1, 2, and 3 are evaluated
    simultaneously by default, covering the full Or-opt-3 neighbourhood in a
    single pass.  The loop terminates when no single relocation improves total
    travel cost.

    Capacity is verified for every inter-route relocation. Intra-route moves
    preserve route loads and are unconditionally feasible.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix ``(N+1, N+1)``, depot at index 0.
        wastes: Mapping from node index to demand (used for capacity checks on
            inter-route relocations).
        capacity: Maximum vehicle load.
        C: Cost per unit distance.  Use ``1.0`` for CVRP distance minimization;
            the actual cost coefficient for VRPP profit weighting.
        chain_lengths: Tuple of chain sizes to evaluate simultaneously.
            Default ``(1, 2, 3)`` covers the full Or-opt-3 neighbourhood.
            Use ``(1,)`` to restrict to single-node relocations.
        max_iter: Safety upper bound on iterations.  In practice the search
            converges well before this limit.

    Returns:
        List[List[int]]: Routes at an Or-opt local minimum.  The input *routes*
            is never mutated — a deep copy is made internally.  Empty routes
            produced by moving all of a route's customers elsewhere are dropped.

    Example:
        >>> improved = or_opt_steepest(routes, dist_matrix, wastes, capacity)
        >>> single_node = or_opt_steepest(routes, dist_matrix, wastes, capacity,
        ...                               chain_lengths=(1,))
    """
    working = [list(r) for r in routes if r]
    if not working:
        return working

    loads = [sum(wastes.get(n, 0.0) for n in r) for r in working]

    for _ in range(max_iter):
        _, move = _find_best_or_opt(working, loads, dist_matrix, wastes, capacity, C, chain_lengths)
        if move is None:
            break

        r1_idx, k, p, r2_idx, q = move
        chain = working[r1_idx][p : p + k]
        chain_load = sum(wastes.get(n, 0.0) for n in chain)

        # Source route after chain removal
        r1_post = working[r1_idx][:p] + working[r1_idx][p + k :]

        if r1_idx == r2_idx:
            # Intra-route: insert chain into the post-removal version
            new_route = r1_post[: q + 1] + chain + r1_post[q + 1 :]
            working[r1_idx] = new_route
            # Load is unchanged; no load update needed
        else:
            # Inter-route: update source and target independently
            working[r1_idx] = r1_post
            working[r2_idx] = working[r2_idx][: q + 1] + chain + working[r2_idx][q + 1 :]
            loads[r1_idx] -= chain_load
            loads[r2_idx] += chain_load

        # Prune empty routes (can occur when the chain was a route's only nodes)
        new_working = [r for r in working if r]
        if len(new_working) != len(working):
            working = new_working
            loads = [sum(wastes.get(n, 0.0) for n in r) for r in working]

    return [r for r in working if r]


def or_opt_steepest_profit(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    chain_lengths: Tuple[int, ...] = (1, 2, 3),
    max_iter: int = 500,
) -> List[List[int]]:
    """Steepest-descent Or-opt for VRPP profit maximisation.

    Delegates to :func:`or_opt_steepest` with ``C`` forwarded.  Revenue is
    unchanged because Or-opt only repositions — never drops — customers.

    Args:
        routes: Current plan.
        dist_matrix: Distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Vehicle capacity.
        R: Revenue per unit waste (unused; revenue unchanged by Or-opt).
        C: Cost per unit distance.
        chain_lengths: Chain sizes to evaluate.
        max_iter: Maximum iterations.

    Returns:
        List[List[int]]: Routes at an Or-opt local minimum.
    """
    return or_opt_steepest(
        routes,
        dist_matrix,
        wastes,
        capacity,
        C=C,
        chain_lengths=chain_lengths,
        max_iter=max_iter,
    )
