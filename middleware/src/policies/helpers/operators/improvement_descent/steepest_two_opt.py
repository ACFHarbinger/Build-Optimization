"""
Steepest-Descent 2-opt Intensification Operator.

Drives a VRP solution to a strict intra-route 2-opt local minimum. At each
iteration the globally best improving edge-reversal move is located across
all routes and applied; the loop terminates once no improving move remains.

This is a pure intensification operator: it never changes which customers
are visited or which route they belong to. Route loads and revenue are
therefore guaranteed to be preserved. For VRPP problems the improvement
threshold is scaled by C so that comparisons remain in profit units.

The complexity per iteration is O(Σ_r n_r²) where n_r is the route length.
In practice the algorithm converges in far fewer iterations than max_iter.

Reference:
    Lin, S. (1965). "Computer solutions of the traveling salesman problem."
    Bell System Technical Journal, 44(10), 2245-2269.

Example:
    >>> from logic.src.policies.helpers.operators.intensification import two_opt_steepest
    >>> improved = two_opt_steepest(routes, dist_matrix, wastes, capacity)
    >>> improved_vrpp = two_opt_steepest_profit(routes, dist_matrix, wastes, capacity, R=1.0, C=0.5)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Internal search engine
# ---------------------------------------------------------------------------


def _find_best_2opt(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    C: float,
) -> Tuple[float, Optional[Tuple[int, int, int]]]:
    """Find the globally best improving intra-route 2-opt move.

    Scans every route for the segment-reversal with the largest improvement.
    Only moves that strictly improve total travel cost by more than 1e-6 are
    considered.  For a symmetric distance matrix, reversing the segment
    ``route[i:j]`` only changes the two boundary edges.

    The reversal replaces edges ``(prev→route[i])`` and ``(route[j-1]→next)``
    with ``(prev→route[j-1])`` and ``(route[i]→next)``.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix, depot at index 0.
        C: Cost coefficient.  Scales the delta for unit consistency; typically
            1.0 for CVRP and the actual cost-per-km for VRPP.

    Returns:
        ``(best_delta, (r_idx, i, j))`` where ``routes[r_idx][i:j]`` should be
        reversed, or ``(0.0, None)`` when no improving move exists.
    """
    best_delta = -1e-6  # Strict improvement threshold
    best_move: Optional[Tuple[int, int, int]] = None
    d = dist_matrix

    for r_idx, route in enumerate(routes):
        n = len(route)
        if n < 3:
            continue  # Need at least 3 nodes for a non-trivial reversal

        for i in range(n - 1):
            a = route[i]
            prev = route[i - 1] if i > 0 else 0

            for j in range(i + 2, n + 1):
                # Full-route reversal (i=0, j=n) is a no-op for symmetric d
                if i == 0 and j == n:
                    continue

                b = route[j - 1]
                nxt = route[j] if j < n else 0

                # Reversing route[i:j] swaps boundary edges:
                #   (prev→a) + (b→nxt)  →  (prev→b) + (a→nxt)
                delta = (d[prev, b] + d[a, nxt] - d[prev, a] - d[b, nxt]) * C

                if delta < best_delta:
                    best_delta = delta
                    best_move = (r_idx, i, j)

    return best_delta, best_move


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def two_opt_steepest(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    C: float = 1.0,
    max_iter: int = 500,
) -> List[List[int]]:
    """Drive the solution to a strict intra-route 2-opt local minimum.

    At every iteration the globally best improving segment-reversal is located
    across all routes and applied. The search terminates once no single 2-opt
    move improves the total travel cost.

    2-opt never changes route membership or the set of visited customers, so
    capacity constraints and revenue are always preserved. For VRPP problems
    set ``C`` to the cost-per-km so that the improvement threshold is correctly
    scaled in profit units; the stopping condition remains identical to CVRP.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix of shape ``(N+1, N+1)``, depot at
            index 0.
        wastes: Node demand lookup (unused; included for API consistency with
            all operators in this package).
        capacity: Maximum vehicle load (unused; 2-opt never moves customers
            between routes, so loads are unchanged).
        C: Cost per unit of distance.  Scales the improvement comparison; use
            ``1.0`` for pure distance minimization (CVRP), or the actual cost
            coefficient for VRPP profit weighting.
        max_iter: Maximum number of improvement iterations.  The search
            typically terminates well before this limit; the bound guards only
            against pathological instances.

    Returns:
        List[List[int]]: Updated routes at a 2-opt local minimum.  The input
            *routes* object is never mutated — a deep copy is made internally.

    Example:
        >>> improved = two_opt_steepest(routes, dist_matrix, wastes, capacity)
        >>> improved = two_opt_steepest(routes, dist_matrix, wastes, capacity, C=0.5)
    """
    working = [list(r) for r in routes if r]
    if not working:
        return working

    for _ in range(max_iter):
        _, move = _find_best_2opt(working, dist_matrix, C)
        if move is None:
            break
        r_idx, i, j = move
        working[r_idx][i:j] = working[r_idx][i:j][::-1]

    return [r for r in working if r]


def two_opt_steepest_profit(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    max_iter: int = 500,
) -> List[List[int]]:
    """Steepest-descent 2-opt for VRPP profit maximisation.

    Delegates to :func:`two_opt_steepest` with ``C`` forwarded.  Revenue is
    unchanged by any 2-opt move (same customers in the same routes, just
    different sequence), so only the distance component of the profit objective
    changes.  The stopping condition is therefore identical to CVRP 2-opt.

    Args:
        routes: Current plan (list of customer sequences, depot implicit at 0).
        dist_matrix: Square distance matrix (depot at index 0).
        wastes: Node demand lookup (unused by 2-opt).
        capacity: Vehicle capacity (unused by 2-opt).
        R: Revenue per unit waste (unused; revenue unchanged by 2-opt).
        C: Cost per unit distance — used to scale the improvement threshold.
        max_iter: Maximum improvement iterations.

    Returns:
        List[List[int]]: Routes at a 2-opt local minimum.
    """
    return two_opt_steepest(routes, dist_matrix, wastes, capacity, C=C, max_iter=max_iter)
