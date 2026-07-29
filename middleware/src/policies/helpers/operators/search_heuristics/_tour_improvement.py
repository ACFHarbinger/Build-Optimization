"""
LKH Tour Improvement Components.

Implements the sequential k-opt local-search loop (k = 2..5) used
in the Lin-Kernighan-Helsgaun heuristic (Helsgaun 2000, Sections 3–4).

These routines iterate over potential cut positions and use exact gain
pre-screening before delegating the actual segment swaps to the shared
:func:`move_kopt_intra` operator.

The high-level driver for these components is :func:`_improve_tour` in
``lin_kernighan_helsgaun.py``.

Architecture
------------
The improvement routines are granular:

1. **_apply_kopt_via_operator** — lowest-level bridge.  Wraps a tour in a
   :class:`TourAdapter`, calls ``move_kopt_intra(k=k)``, and returns the
   updated closed tour if the operator applied a move.

2. **_try_2opt_move** — 2-opt search over α-nearest neighbours of t2.
   Uses :func:`_2opt_gain` as a pre-screen so the operator is only called
   when the exact gain is positive.

3. **_try_3opt_move** — 3-opt search over all valid third cut positions.
   Uses :func:`_3opt_gains` (seven cases) as a pre-screen; cases 1 and 2
   (pure 2-opt sub-moves) are excluded to avoid duplicating the 2-opt phase.

4. **_try_4opt_move** — 4-opt search over all valid fourth cut positions.
   Uses :func:`_4opt_gains` (three cases) as a pre-screen.

5. **_try_5opt_move** — 5-opt search over all valid fifth cut positions.
   Uses :func:`_5opt_gains` (five cases) as a pre-screen. This is the
   "basic move" of Helsgaun (2000), Section 4.3.

Instance-size thresholds
------------------------
Higher-order moves scale super-linearly in the number of cut positions, so
they are restricted by route length:

- 3-opt: n < 500
- 4-opt: n < 300
- 5-opt: n < 200

Dependencies
------------
- :func:`get_cost`, :func:`is_better` from ``._objective``
- :class:`TourAdapter` from ``._tour_adapter``
- :func:`_2opt_gain`, :func:`_3opt_gains`, :func:`_4opt_gains`,
  :func:`_5opt_gains` from ``._tour_construction``
- :func:`move_kopt_intra` from the shared intra-route operator package

Typical usage
-------------
>>> from logic.src.policies.helpers.operators.heuristics._tour_improvement import (
...     _2opt_gain,               # Exact gain for 2-opt
...     _apply_kopt_via_operator,  # Bridge to move_kopt_intra
...     _try_3opt_move,           # Sequential 3-opt search
...     _try_4opt_move,           # Sequential 4-opt search
...     _try_5opt_move,           # Sequential 5-opt search
... )
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import move_kopt_intra
from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_adapter import TourAdapter
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _2opt_gain,
    _3opt_gains,
    _4opt_gains,
    _5opt_gains,
)

# ---------------------------------------------------------------------------
# k-opt move application via the shared operator
# ---------------------------------------------------------------------------


def _apply_kopt_via_operator(
    tour: List[int],
    p_u: int,
    p_v: int,
    k: int,
    distance_matrix: np.ndarray,
    rng: Random,
) -> Optional[List[int]]:
    """
    Attempt a k-opt move via :func:`move_kopt_intra` and return the new tour.

    Wraps the tour in a :class:`TourAdapter`, calls ``move_kopt_intra`` with
    the given cut positions, and returns the resulting closed tour only if the
    operator applied an improving move (i.e. ``_update_map`` was called).

    ``move_kopt_intra`` behaviour by k:

    - **k=2**: Checks ``delta * C < -1e-4`` (exact cost improvement).
    - **k=3**: Randomly samples a third cut and evaluates four reconnection
      patterns (g4..g7), applying the best improving one.
    - **k≥4**: Enumerates all permutations × orientations of the middle
      segments and applies the configuration with the greatest saving.

    Args:
        tour: Current closed tour.
        p_u: Position of the first cut node in the open route.
        p_v: Position of the second cut node in the open route.
        k: Number of edges to remove (2, 3, 4, or 5).
        distance_matrix: (n × n) cost matrix.
        rng: Random number generator (required for k ≥ 3).

    Returns:
        The new closed tour if an improvement was applied; ``None`` otherwise.
    """
    adapter = TourAdapter(tour, distance_matrix)
    route = adapter.routes[0]
    n = len(route)

    if p_u < 0 or p_v < 0 or p_u >= n or p_v >= n or p_u == p_v:
        return None

    u = route[p_u]
    v = route[p_v]

    applied = move_kopt_intra(
        adapter,
        u=u,
        v=v,
        r_u=0,
        p_u=p_u,
        r_v=0,
        p_v=p_v,
        k=k,
        rng=rng if k >= 3 else None,
    )

    if applied:
        return adapter.to_closed_tour()
    return None


# ---------------------------------------------------------------------------
# Inner search routines with gain pre-screening
# ---------------------------------------------------------------------------


def _try_2opt_move(
    curr_tour: List[int],
    i: int,
    t1: int,
    t2: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    rng: Random,
    pos_map: Dict[int, int],
) -> Tuple[Optional[List[int]], float, bool, int]:
    """
    Search for an improving 2-opt move starting from edge (t1, t2).

    Iterates over α-nearest neighbours of t2 as the candidate for t3.
    Uses an exact gain pre-screen (:func:`_2opt_gain`) to skip clearly
    non-improving pairs, then delegates the actual move to
    :func:`move_kopt_intra` (k=2) via :func:`_apply_kopt_via_operator`.

    Args:
        curr_tour: Current closed tour.
        i: Position of t1 in the open route.
        t1, t2: Endpoints of the edge being considered for removal.
        candidates: α-nearest-neighbour lists.
        distance_matrix: Cost matrix.
        rng: Random number generator.
        pos_map: $O(1)$ position map for tour index lookups.

    Returns:
        (new_tour, cost, improved, j) where j is the position of t3.
        Returns (None, 0.0, False, -1) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    for t3 in candidates[t2]:
        if t3 == t1:
            continue
        if t3 == curr_tour[(i + 2) % nodes_count]:
            continue

        j = pos_map.get(t3, -1)
        if j == -1 or j <= i + 1:
            continue

        t4 = curr_tour[j + 1]

        # Exact gain pre-screen: only call the operator when gain > 0
        gain = _2opt_gain(t1, t2, t3, t4, d)
        if gain > 1e-6:
            new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
            if new_tour is not None:
                c_new = get_cost(new_tour, d)
                return new_tour, c_new, True, j

    return None, 0.0, False, -1


def _try_3opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    distance_matrix: np.ndarray,
    rng: Random,
) -> Tuple[Optional[List[int]], float, bool]:
    """
    Search for an improving 3-opt move extending the (t1,t2), (t3,t4) pair.

    For each valid third cut position k_pos (t5, t6), computes exact gains
    (:func:`_3opt_gains`) for all seven non-trivial reconnection cases.
    Applies the best improving reconnection directly using deterministic list
    slicing to avoid the operator disconnect.

    Cases 1 and 2 (pure 2-opt sub-moves) are excluded to avoid redundancy
    with the 2-opt phase.

    Args:
        curr_tour: Current closed tour.
        i, j: Break-point positions of the first two cuts in the open route.
        t1..t4: Node pairs for the two existing broken edges.
        distance_matrix: Cost matrix.
        rng: Random number generator (unused, kept for API compatibility).

    Returns:
        (new_tour, cost, improved).
        Returns (None, 0.0, False) if no improvement found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_c = get_cost(curr_tour, d)

    for k_pos in range(j + 2, nodes_count):
        t5 = curr_tour[k_pos]
        t6 = curr_tour[k_pos + 1]

        gains = _3opt_gains(t1, t2, t3, t4, t5, t6, d)

        # Cases 1 & 2 are pure 2-opt; skip them
        best_gain = 1e-6
        best_case = -1

        for idx, g in enumerate(gains):
            if idx not in (1, 2) and g > best_gain:
                best_gain = g
                best_case = idx

        if best_case != -1:
            # Reconstruct tour via direct slicing
            # Segments: A = tour[:i+1], B = tour[i+1:j+1], C = tour[j+1:k_pos+1], D = tour[k_pos+1:]
            A = curr_tour[: i + 1]
            B = curr_tour[i + 1 : j + 1]
            C = curr_tour[j + 1 : k_pos + 1]
            D = curr_tour[k_pos + 1 :]

            if best_case == 0:
                new_tour = A + B[::-1] + C[::-1] + D
            elif best_case == 3:
                new_tour = A + C + B + D
            elif best_case == 4:
                new_tour = A + C + B[::-1] + D
            elif best_case == 5:
                new_tour = A + C[::-1] + B + D
            elif best_case == 6:
                new_tour = A + C[::-1] + B[::-1] + D
            else:
                continue

            c3 = get_cost(new_tour, d)
            if is_better(c3, curr_c):
                return new_tour, c3, True

    return None, 0.0, False


def _try_4opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    distance_matrix: np.ndarray,
    rng: Random,
) -> Tuple[Optional[List[int]], float, bool]:
    """
    Search for an improving 4-opt move extending three broken edges.

    For each valid fourth cut position l (t7, t8), computes exact gains
    (:func:`_4opt_gains`) for three common 4-opt patterns. Applies the best
    improving reconnection directly using deterministic list slicing to avoid
    the operator disconnect.

    Args:
        curr_tour: Current closed tour.
        i, j, k: Break-point positions of the first three cuts.
        t1..t6: Node pairs for the three existing broken edges.
        distance_matrix: Cost matrix.
        rng: Random number generator (unused, kept for API compatibility).

    Returns:
        (new_tour, cost, improved).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_c = get_cost(curr_tour, d)

    for l in range(k + 2, nodes_count):
        t7 = curr_tour[l]
        t8 = curr_tour[l + 1]

        gains = _4opt_gains(t1, t2, t3, t4, t5, t6, t7, t8, d)

        # Find best improving case
        best_gain = 1e-6
        best_case = -1

        for idx, g in enumerate(gains):
            if g > best_gain:
                best_gain = g
                best_case = idx

        if best_case != -1:
            # Reconstruct tour via direct slicing
            # Segments: A=tour[:i+1], B=tour[i+1:j+1], C=tour[j+1:k+1], D=tour[k+1:l+1], E=tour[l+1:]
            A = curr_tour[: i + 1]
            B = curr_tour[i + 1 : j + 1]
            C = curr_tour[j + 1 : k + 1]
            D = curr_tour[k + 1 : l + 1]
            E = curr_tour[l + 1 :]

            if best_case == 0:
                # case 0: double-bridge A+C+B+D+E → t1-t5, t6-t3, t4-t7, t2-t8
                # A(ending t1) + C(t5..t6) + B(reversed: t3..t2) + D(t7..t8) + E
                new_tour = A + C + B[::-1] + D + E
            elif best_case == 1:
                # case 1: reverse B and D → t1-t3, t2-t4, t5-t7, t6-t8
                # A(ending t1) + B(reversed t3..t2) + C(t5..t6) + D(reversed t7..t6) + E
                new_tour = A + B[::-1] + C + D[::-1] + E
            elif best_case == 2:
                # case 2: swap B and D → t1-t7, t8-t5, t6-t3, t4-t2
                # A(ending t1) + D(reversed t7..t6) + C(reversed t5..t4) + B(reversed t3..t2) + E
                new_tour = A + D[::-1] + C[::-1] + B[::-1] + E
            else:
                continue

            c4 = get_cost(new_tour, d)
            if is_better(c4, curr_c):
                return new_tour, c4, True

    return None, 0.0, False


def _try_5opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    l: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    distance_matrix: np.ndarray,
    rng: Random,
) -> Tuple[Optional[List[int]], float, bool]:
    """
    Search for an improving 5-opt move extending four broken edges.

    Implements Helsgaun (2000) Section 4.3: the basic LKH move is a
    sequential 5-opt. For each valid fifth cut position m (t9, t10), exact
    gains (:func:`_5opt_gains`) are computed. Applies the best improving
    reconnection directly using deterministic list slicing to avoid the
    operator disconnect.

    Args:
        curr_tour: Current closed tour.
        i, j, k, l: Break-point positions of the first four cuts.
        t1..t8: Node pairs for the four existing broken edges.
        distance_matrix: Cost matrix.
        rng: Random number generator (unused, kept for API compatibility).

    Returns:
        (new_tour, cost, improved).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_c = get_cost(curr_tour, d)

    for m in range(l + 2, nodes_count):
        t9 = curr_tour[m]
        t10 = curr_tour[m + 1]

        gains = _5opt_gains(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, d)

        # Find best improving case
        best_gain = 1e-6
        best_case = -1

        for idx, g in enumerate(gains):
            if g > best_gain:
                best_gain = g
                best_case = idx

        if best_case != -1:
            # Reconstruct tour via direct slicing
            # Segments: A=tour[:i+1], B=tour[i+1:j+1], C=tour[j+1:k+1], D=tour[k+1:l+1], E=tour[l+1:m+1], F=tour[m+1:]
            A = curr_tour[: i + 1]
            B = curr_tour[i + 1 : j + 1]
            C = curr_tour[j + 1 : k + 1]
            D = curr_tour[k + 1 : l + 1]
            E = curr_tour[l + 1 : m + 1]
            F = curr_tour[m + 1 :]

            if best_case == 0:
                # case 0: A + B' + C + D' + E + F → t1-t3, t2-t4, t5-t7, t6-t8, t9-t10
                new_tour = A + B[::-1] + C + D[::-1] + E + F
            elif best_case == 1:
                # case 1: A + C + B' + D + E + F → t1-t5, t6-t3, t4-t2, t7-t8, t9-t10
                new_tour = A + C + B[::-1] + D + E + F
            elif best_case == 2:
                # case 2: A + B + D + C' + E + F → t1-t2, t3-t7, t8-t5, t6-t4, t9-t10
                new_tour = A + B + D + C[::-1] + E + F
            elif best_case == 3:
                # case 3: A + C + D + B + E + F → t1-t5, t6-t7, t8-t3, t4-t2, t9-t10
                new_tour = A + C + D + B + E + F
            elif best_case == 4:
                # case 4: A + D + B' + C + E + F → t1-t7, t8-t3, t4-t5, t6-t2, t9-t10
                new_tour = A + D + B[::-1] + C + E + F
            else:
                continue

            c5 = get_cost(new_tour, d)
            if is_better(c5, curr_c):
                return new_tour, c5, True

    return None, 0.0, False
