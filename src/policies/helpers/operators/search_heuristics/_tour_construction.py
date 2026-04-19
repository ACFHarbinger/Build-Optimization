"""
LKH-3 Tour Construction and Gain-Computation Module.

Provides tour initialisation, elite-tour merging, and the double-bridge
perturbation used by the LKH-3 iterated local-search loop, together with
exact gain formulae for 2-, 3-, 4-, and 5-opt moves.

Tour construction
-----------------
_initialize_tour(distance_matrix, initial_tour) -> List[int]
    Return a closed starting tour.  If *initial_tour* is ``None`` a greedy
    nearest-neighbour construction heuristic is used (Helsgaun 2000,
    Section 4.4); otherwise the supplied tour is normalised to closed form.

merge_tours(tour1, tour2, distance_matrix) -> List[int]
    Pool-based recombination (Helsgaun 2000, Section 4.4).  Build a new tour
    by preferring edges shared by both inputs; fall back to nearest-unvisited
    selection whenever no shared edge is available.

Perturbation
------------
_double_bridge_kick(tour, distance_matrix, rng) -> List[int]
    Apply a double-bridge (random non-sequential 4-opt) perturbation by
    delegating to :func:`double_bridge` from the shared perturbation operator.
    The move cuts the route at three random positions and reconnects as
    A + C + B + D, producing a tour unreachable by any 2-opt or 3-opt move
    (Helsgaun 2000, Section 3.2 / ILS literature).

Exact gain functions
--------------------
The gain functions compute the net reduction in tour length if the named
edges are removed and the segments are reconnected in the specified pattern.
A positive value means the move is improving.

_2opt_gain(t1, t2, t3, t4, d) -> float
    Gain for removing (t1,t2) and (t3,t4):
    ``c(t1,t2) + c(t3,t4) − c(t1,t3) − c(t2,t4)``.

_3opt_gains(t1, t2, t3, t4, t5, t6, d) -> List[float]
    Gains for all seven non-trivial 3-opt reconnection patterns (cases 0–6).

_4opt_gains(t1, …, t8, d) -> List[float]
    Gains for three common 4-opt reconnection patterns (cases 0–2).

_5opt_gains(t1, …, t10, d) -> List[float]
    Gains for five common 5-opt reconnection patterns (cases 0–4), covering
    the sequential 5-opt moves central to Helsgaun (2000), Section 4.3.

Dependencies
------------
- :class:`TourAdapter` from ``._tour_adapter`` to wrap tours for operator calls.
- :func:`double_bridge` from the shared perturbation operator package.

Typical usage
-------------
>>> from logic.src.policies.helpers.operators.heuristics._tour_construction import (
...     _initialize_tour, _double_bridge_kick, _2opt_gain, _3opt_gains
... )
>>> tour = _initialize_tour(dist, initial_tour=None)
>>> gain = _2opt_gain(t1, t2, t3, t4, dist)
>>> if gain > 1e-6:
...     tour = _double_bridge_kick(tour, dist, rng)
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.operators.perturbation_shaking.double_bridge import double_bridge
from logic.src.policies.helpers.operators.search_heuristics._tour_adapter import TourAdapter

# ---------------------------------------------------------------------------
# Tour construction and perturbation
# ---------------------------------------------------------------------------


def merge_tours(
    tour1: List[int],
    tour2: List[int],
    distance_matrix: np.ndarray,
) -> List[int]:
    """
    Combine two elite tours by preferring edges common to both.

    Shared edges are followed greedily; when no shared edge is available
    the nearest unvisited node is selected.  This implements the pool-based
    recombination described in Helsgaun (2000), Section 4.4.

    Args:
        tour1: First tour (closed).
        tour2: Second tour (closed).
        distance_matrix: (n × n) cost matrix.

    Returns:
        A new closed tour that reuses as many shared edges as possible.
    """
    n = len(tour1) - 1

    edges1: set = set()
    edges2: set = set()
    for i in range(len(tour1) - 1):
        a, b = tour1[i], tour1[i + 1]
        edges1.add((min(a, b), max(a, b)))
    for i in range(len(tour2) - 1):
        a, b = tour2[i], tour2[i + 1]
        edges2.add((min(a, b), max(a, b)))

    common_edges = edges1 & edges2

    if not common_edges:
        cost1 = sum(distance_matrix[tour1[i], tour1[i + 1]] for i in range(len(tour1) - 1))
        cost2 = sum(distance_matrix[tour2[i], tour2[i + 1]] for i in range(len(tour2) - 1))
        return tour1 if cost1 <= cost2 else tour2

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in common_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = [False] * n
    merged = [0]
    visited[0] = True
    current = 0
    while len(merged) < n:
        next_node = None
        for neighbor in adj[current]:
            if not visited[neighbor]:
                next_node = neighbor
                break
        if next_node is None:
            min_dist = float("inf")
            for node in range(n):
                if not visited[node] and distance_matrix[current, node] < min_dist:
                    min_dist = distance_matrix[current, node]
                    next_node = node
        if next_node is None:
            break
        merged.append(next_node)
        visited[next_node] = True
        current = next_node

    merged.append(0)
    return merged


def _double_bridge_kick(
    tour: List[int],
    distance_matrix: np.ndarray,
    rng: Random,
) -> List[int]:
    """
    Apply a double-bridge (random 4-opt) perturbation via the shared operator.

    Delegates to :func:`double_bridge` from
    ``logic.src.policies.helpers.operators.perturbation.double_bridge``, which
    slices the route at three random cut points and reconnects as A+C+B+D.

    A :class:`TourAdapter` bridges the flat tour representation used by
    this solver with the ``ls``-style interface expected by the operator.

    Args:
        tour: Current closed tour (first node repeated at end).
        distance_matrix: (n × n) cost matrix (forwarded to the adapter).
        rng: ``random.Random`` instance for reproducible cut selection.

    Returns:
        New perturbed closed tour.  Returns the original tour unchanged if
        the route is too short for a double-bridge move (< 4 nodes).
    """
    adapter = TourAdapter(tour, distance_matrix)
    applied = double_bridge(adapter, r_idx=0, rng=rng)
    if not applied:
        return tour
    return adapter.to_closed_tour()


def _initialize_tour(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]],
) -> List[int]:
    """
    Build or validate the starting tour.

    If no tour is provided a nearest-neighbour construction heuristic is used
    (Helsgaun 2000, Section 4.4).  The returned tour is always closed (first
    node repeated at end).

    Args:
        distance_matrix: (n × n) cost matrix.
        initial_tour: Optional existing tour.

    Returns:
        A closed tour over all n nodes.
    """
    n = len(distance_matrix)
    if initial_tour is None:
        curr = 0
        path = [0]
        unvisited = set(range(1, n))
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
            path.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        path.append(0)
        return path
    else:
        t = initial_tour[:]
        if t[0] != t[-1]:
            t.append(t[0])
        return t


# ---------------------------------------------------------------------------
# Exact gain computations for each k-opt level
# ---------------------------------------------------------------------------


def _2opt_gain(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    d: np.ndarray,
) -> float:
    """
    Exact gain for a 2-opt move that removes (t1,t2) and (t3,t4).

    Gain = c(t1,t2) + c(t3,t4) − c(t1,t3) − c(t2,t4).

    A positive gain means the 2-opt move strictly reduces tour length.
    """
    return d[t1, t2] + d[t3, t4] - d[t1, t3] - d[t2, t4]


def _3opt_gains(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    d: np.ndarray,
) -> List[float]:
    """
    Exact gains for all seven non-trivial 3-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6).  The original tour cost
    contribution is d(t1,t2) + d(t3,t4) + d(t5,t6).

    Returns a list of seven gain values corresponding to cases 0–6 of
    ``move_kopt_intra`` (k=3).

    Gain > 0 means an improving move for that case.
    """
    base = d[t1, t2] + d[t3, t4] + d[t5, t6]
    return [
        # case 0: A + B' + C' + D  →  t1-t3, t2-t5, t4-t6
        base - (d[t1, t3] + d[t2, t5] + d[t4, t6]),
        # case 1: A + B  + C' + D  →  pure reversal of C (2-opt sub-move)
        base - (d[t1, t2] + d[t3, t5] + d[t4, t6]),
        # case 2: A + B' + C  + D  →  pure reversal of B (2-opt sub-move)
        base - (d[t1, t3] + d[t2, t4] + d[t5, t6]),
        # case 3: A + C  + B  + D  →  t1-t4, t5-t2, t3-t6
        base - (d[t1, t4] + d[t5, t2] + d[t3, t6]),
        # case 4: A + C  + B' + D  →  t1-t4, t5-t3, t2-t6
        base - (d[t1, t4] + d[t5, t3] + d[t2, t6]),
        # case 5: A + C' + B  + D  →  t1-t5, t4-t2, t3-t6
        base - (d[t1, t5] + d[t4, t2] + d[t3, t6]),
        # case 6: A + C' + B' + D  →  t1-t5, t4-t3, t2-t6
        base - (d[t1, t5] + d[t4, t3] + d[t2, t6]),
    ]


def _4opt_gains(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    d: np.ndarray,
) -> List[float]:
    """
    Exact gains for three common 4-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6), (t7,t8).

    Returns a list of three gain values (indices 0–2).
    """
    base = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8]
    return [
        # case 0: double-bridge  A+C+B+D+E  →  t1-t5, t6-t3, t4-t7, t2-t8
        base - (d[t1, t5] + d[t6, t3] + d[t4, t7] + d[t2, t8]),
        # case 1: reverse B and D  →  t1-t3, t2-t4, t5-t7, t6-t8
        base - (d[t1, t3] + d[t2, t4] + d[t5, t7] + d[t6, t8]),
        # case 2: swap B and D  →  t1-t7, t8-t5, t6-t3, t4-t2
        base - (d[t1, t7] + d[t8, t5] + d[t6, t3] + d[t4, t2]),
    ]


def _5opt_gains(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    t9: int,
    t10: int,
    d: np.ndarray,
) -> List[float]:
    """
    Exact gains for five common 5-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6), (t7,t8), (t9,t10).

    Returns a list of five gain values (indices 0–4).
    """
    base = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8] + d[t9, t10]
    return [
        # case 0: A + B' + C + D' + E + F
        base - (d[t1, t3] + d[t2, t4] + d[t5, t7] + d[t6, t8] + d[t9, t10]),
        # case 1: A + C + B' + D + E + F
        base - (d[t1, t5] + d[t6, t3] + d[t4, t2] + d[t7, t8] + d[t9, t10]),
        # case 2: A + B + D + C' + E + F
        base - (d[t1, t2] + d[t3, t7] + d[t8, t5] + d[t6, t4] + d[t9, t10]),
        # case 3: A + C + D + B + E + F
        base - (d[t1, t5] + d[t6, t7] + d[t8, t3] + d[t4, t2] + d[t9, t10]),
        # case 4: A + D + B' + C + E + F
        base - (d[t1, t7] + d[t8, t3] + d[t4, t5] + d[t6, t2] + d[t9, t10]),
    ]
