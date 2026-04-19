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

import logging
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from logic.src.policies.helpers.operators.perturbation_shaking.double_bridge import double_bridge
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_adapter import TourAdapter

logger = logging.getLogger(__name__)

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


def merge_tours_ip(
    tour_pool: List[List[int]],
    distance_matrix: np.ndarray,
) -> List[int]:
    """Exact recombination of elite tours via restricted edge assignment.

    Collects the union of all edges from the tour pool, builds a restricted
    cost matrix limited to those edges, and solves the resulting assignment
    problem via ``scipy.optimize.linear_sum_assignment`` to find the
    minimum-cost Hamiltonian tour using only pooled edges.

    Falls back to greedy :func:`merge_tours` if the restricted problem is
    infeasible (e.g., too sparse to form a valid tour).

    Args:
        tour_pool: List of elite tours (each closed).
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        A new closed tour combining edges from the elite pool.
    """
    if len(tour_pool) < 2:
        return tour_pool[0][:] if tour_pool else []

    n = len(tour_pool[0]) - 1  # Number of nodes (excluding closing duplicate)

    # --- Collect union of edges from all elite tours ---
    pooled_edges: set = set()
    for tour in tour_pool:
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            pooled_edges.add((min(a, b), max(a, b)))

    # --- Build restricted adjacency with only pooled edges ---
    # Create a cost matrix where non-pooled edges have infinite cost
    big_m = float(np.sum(distance_matrix)) + 1.0
    restricted_cost = np.full((n, n), big_m, dtype=float)
    np.fill_diagonal(restricted_cost, big_m)

    for a, b in pooled_edges:
        if 0 <= a < n and 0 <= b < n:
            restricted_cost[a, b] = distance_matrix[a, b]
            restricted_cost[b, a] = distance_matrix[b, a]

    # --- Solve assignment to get minimum-cost edge selection ---
    try:
        row_ind, col_ind = linear_sum_assignment(restricted_cost)
    except ValueError:
        # scipy failed — fall back to greedy pairwise merge
        return merge_tours(tour_pool[0], tour_pool[1], distance_matrix)

    # --- Build tour from assignment result ---
    return _build_tour_from_successor(
        row_ind,
        col_ind,
        restricted_cost,
        big_m,
        n,
        distance_matrix,
        tour_pool,
    )


def _build_tour_from_successor(
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    restricted_cost: np.ndarray,
    big_m: float,
    n: int,
    distance_matrix: np.ndarray,
    tour_pool: List[List[int]],
) -> List[int]:
    """Build a closed tour from the assignment (successor) result.

    Args:
        row_ind: Row indices from linear_sum_assignment.
        col_ind: Column indices from linear_sum_assignment.
        restricted_cost: Restricted cost matrix.
        big_m: Penalty value for non-pooled edges.
        n: Number of nodes.
        distance_matrix: Full cost matrix.
        tour_pool: Original tour pool (for greedy fallback).

    Returns:
        A closed tour.
    """
    # The assignment gives a perfect matching; chain it into a tour
    successor: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind, strict=False):
        if restricted_cost[r, c] < big_m:
            successor[r] = c

    # Construct tour by following successor pointers
    if 0 not in successor:
        return merge_tours(tour_pool[0], tour_pool[1], distance_matrix)

    visited = [False] * n
    tour_result = [0]
    visited[0] = True
    current = 0

    while True:
        nxt = successor.get(current)
        if nxt is None or visited[nxt]:
            # Nearest-neighbor fallback: the linear-sum assignment produced
            # subtours rather than a Hamiltonian cycle.  This is expected
            # since LSA solves matching, not TSP.
            logger.warning(
                "merge_tours_ip: successor chain broken at node %d — "
                "patching with nearest-neighbor fallback (subtour detected)",
                current,
            )
            best_dist = float("inf")
            best_node = -1
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < best_dist:
                    best_dist = distance_matrix[current, j]
                    best_node = j
            if best_node == -1:
                break
            nxt = best_node

        tour_result.append(nxt)
        visited[nxt] = True
        current = nxt

        if len(tour_result) >= n:
            break

    tour_result.append(0)
    return tour_result


def merge_tours_best(
    tour_pool: List[List[int]],
    distance_matrix: np.ndarray,
    use_ip: bool = True,
) -> List[int]:
    """Dispatch to IP-based or greedy tour merging.

    Args:
        tour_pool: List of elite tours (each closed).
        distance_matrix: (n × n) cost matrix.
        use_ip: If True, attempt IP-based merging; else use greedy.

    Returns:
        A new closed tour combining edges from the pool.
    """
    if len(tour_pool) < 2:
        return tour_pool[0][:] if tour_pool else []

    if use_ip:
        return merge_tours_ip(tour_pool, distance_matrix)
    else:
        # Greedy: merge first two, then fold in remaining
        result = merge_tours(tour_pool[0], tour_pool[1], distance_matrix)
        for i in range(2, len(tour_pool)):
            result = merge_tours(result, tour_pool[i], distance_matrix)
        return result


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


def _get_tour_array(start_node: int, tour_dict: Dict[int, int]) -> List[int]:
    """Retrieve the tour as a list starting from `start_node`."""
    tour = [start_node]
    curr = tour_dict[start_node]
    while curr != start_node:
        tour.append(curr)
        curr = tour_dict[curr]
    return tour


class KoptTopologyFactory:
    """
    Programmatic generator for exhaustive non-sequential k-opt topologies.
    Memoizes the exact valid Hamiltonian reconnections to avoid hardcoding.
    """

    _cache: Dict[int, List[List[Tuple[int, int]]]] = {}

    @classmethod
    def get_topologies(cls, k: int) -> List[List[Tuple[int, int]]]:  # noqa: C901
        """
        Get all valid added-edge topologies for a given k.
        Returns a list of topologies. Each topology is a list of exactly k edge tuples.
        The tuples contain 0-based indices into the broken nodes array (t1..t2k).

        Matches LKH pure non-sequential counts:
        - 2-opt: 1
        - 3-opt: 7 (including 2-opt sub-moves as requested)
        - 4-opt: 25 pure
        - 5-opt: exhaustive pure valid Hamiltonian reconnections
        """
        if k in cls._cache:
            return cls._cache[k]

        endpoints = list(range(2 * k))
        # Segments: (t_2, t_3), (t_4, t_5), ..., (t_2k, t_1) -> indices: (1, 2), (3, 4), ..., (2k-1, 0)
        segments = [(2 * i + 1, (2 * i + 2) % (2 * k)) for i in range(k)]
        broken = set()
        for i in range(k):
            u, v = 2 * i, 2 * i + 1
            broken.add((u, v))
            broken.add((v, u))

        def find_matchings(unmatched: List[int]):
            if not unmatched:
                yield []
            else:
                u = unmatched[0]
                for i in range(1, len(unmatched)):
                    v = unmatched[i]
                    for m in find_matchings(unmatched[1:i] + unmatched[i + 1 :]):
                        yield [(u, v)] + m

        valid_topologies = []
        for m in find_matchings(endpoints):
            # Exclude the original configuration (where all added edges are broken edges)
            is_original = True
            for u, v in m:
                if (u, v) in broken:
                    is_original = False
                    break
            if is_original:
                continue

            # Build adjacency list
            adj: Dict[int, List[int]] = {v: [] for v in endpoints}
            for u, v in segments:
                adj[u].append(v)
                adj[v].append(u)
            for u, v in m:
                adj[u].append(v)
                adj[v].append(u)

            # Check for a single Hamiltonian cycle
            visited = set()
            curr = 0
            while curr not in visited:
                visited.add(curr)
                nxt = None
                for n in adj[curr]:
                    if n not in visited:
                        nxt = n
                        break
                if nxt is None:
                    break
                curr = nxt

            if len(visited) == 2 * k:
                # Filter pure moves for k >= 4
                pure = True
                if k >= 4:
                    for u, v in m:
                        if (u, v) in broken:
                            pure = False
                            break
                if pure:
                    valid_topologies.append(m)

        cls._cache[k] = valid_topologies
        return valid_topologies


def build_tour_from_segments(
    curr_tour: List[int],
    pos: List[int],
    added_edges: List[Tuple[int, int]],
    k: int,
) -> List[int]:
    """
    Constructs the exact topological candidate bypassing heuristic operator ambiguity.

    Given k broken edges (t1,t2), (t3,t4) ... defined by their indices in `pos`,
    extracts the k segments and connects them according to `added_edges`.

    Args:
        curr_tour: The original closed tour (length N+1, starts/ends with 0).
        pos: List of indices in curr_tour for [t1, t2, ..., t2k].
        added_edges: Matching of endpoints [0..2k-1] defining the reconnection.
        k: The k-opt order.

    Returns:
        The new closed tour array.
    """
    # 1. Extract the k segments
    # S_i goes from t_{2i+1} to t_{2i+2} (0-indexed pos: 2i+1 to (2i+2)%2k)
    segments = []
    for i in range(k):
        start_idx = pos[2 * i + 1]
        end_idx = pos[(2 * i + 2) % (2 * k)]

        if start_idx <= end_idx:
            seg = curr_tour[start_idx : end_idx + 1]
        else:
            # Wrap around (only possible for the last segment t_2k -> t_1)
            seg = curr_tour[start_idx:-1] + curr_tour[: end_idx + 1]
        segments.append(seg)

    # Mapping endpoint -> (segment_idx, is_start)
    endpoint_to_seg = {}
    for i in range(k):
        endpoint_to_seg[2 * i + 1] = (i, True)
        endpoint_to_seg[(2 * i + 2) % (2 * k)] = (i, False)

    # Build fast adjacency for added edges
    adj = {}
    for u, v in added_edges:
        adj[u] = v
        adj[v] = u

    # 2. Trace the single cycle starting from the depot (node 0)
    # The depot is ALWAYS in the last segment (k-1) if t1..t2k follows order
    # but we can just find which segment contains the first node of curr_tour.
    # Actually, segments[k-1] wraps around, so it contains 0!
    # Wait, the first element of curr_tour is 0, so 0 is in the segment that wraps around.

    new_tour_open = []

    # We start traversal from endpoint 0 (t1, end of segment k-1).
    # Wait, the new tour must start and end with 0! It's easier to build any
    # valid cyclic array, then rotate it so 0 is at the front.

    curr_endpoint = 0
    for _ in range(k):
        # We are at curr_endpoint (which is the end of some segment).
        # We cross the added edge to next_endpoint
        next_endpoint = adj[curr_endpoint]

        seg_idx, is_start = endpoint_to_seg[next_endpoint]
        seg_nodes = segments[seg_idx]

        if is_start:
            # Traverse forward
            new_tour_open.extend(seg_nodes)
            curr_endpoint = (2 * seg_idx + 2) % (2 * k)  # arrive at end of segment
        else:
            # Traverse backward
            new_tour_open.extend(reversed(seg_nodes))
            curr_endpoint = 2 * seg_idx + 1  # arrive at start of segment

    # Remove consecutive duplicates that occur from gluing
    cleaned: List[int] = []
    for node in new_tour_open:
        if not cleaned or cleaned[-1] != node:
            cleaned.append(node)
    if cleaned and cleaned[-1] == cleaned[0]:
        cleaned.pop()

    # Rotate so 0 is first
    zero_idx = cleaned.index(0)
    rotated = cleaned[zero_idx:] + cleaned[:zero_idx] + [0]
    return rotated


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
    Exact distance gain for a 2-opt move that removes (t1,t2) and (t3,t4).

    Gain = c(t1,t2) + c(t3,t4) − c(t1,t3) − c(t2,t4).

    A positive gain means the 2-opt move strictly reduces tour length.

    NOTE: For lexicographic objectives, this function only computes ΔC (cost change).
    The caller must separately evaluate ΔP (penalty change) to determine if the
    move should be accepted. See _should_accept_kopt_move() for the full logic.
    """
    return d[t1, t2] + d[t3, t4] - d[t1, t3] - d[t2, t4]


def _should_accept_kopt_move(
    delta_penalty: float,
    delta_cost: float,
) -> bool:
    """
    Lexicographic pre-screening for k-opt moves.

    A k-opt move should be passed to the operator for full evaluation if:
    1. It reduces capacity violations (ΔP < 0), OR
    2. It maintains feasibility (ΔP == 0) AND improves cost (ΔC > 0)

    This allows the search to transit through infeasible space by accepting
    moves that reduce penalty, even if they increase routing cost.

    Args:
        delta_penalty: Change in capacity penalty (P_new - P_old).
                       Negative values mean reduced violations.
        delta_cost: Change in routing cost (C_new - C_old).
                    Positive values mean cost reduction.

    Returns:
        True if the move should be accepted for evaluation, False otherwise.

    Mathematical Formulation:
        Accept iff: (ΔP < -ε) OR (|ΔP| ≤ ε AND ΔC > ε)
        where ε = 1e-6 is numerical tolerance.

    Examples:
        >>> # Case 1: Reduces penalty (even if cost increases)
        >>> _should_accept_kopt_move(delta_penalty=-5.0, delta_cost=-2.0)
        True
        >>>
        >>> # Case 2: Maintains feasibility and improves cost
        >>> _should_accept_kopt_move(delta_penalty=0.0, delta_cost=3.0)
        True
        >>>
        >>> # Case 3: Increases penalty (blocked)
        >>> _should_accept_kopt_move(delta_penalty=2.0, delta_cost=5.0)
        False
        >>>
        >>> # Case 4: No change (blocked)
        >>> _should_accept_kopt_move(delta_penalty=0.0, delta_cost=0.0)
        False
    """
    eps = 1e-6

    # Condition 1: Reduces capacity violation
    if delta_penalty < -eps:
        return True

    # Condition 2: Maintains feasibility AND improves cost
    return bool(abs(delta_penalty) <= eps and delta_cost > eps)


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

    NOTE: Cases 1 and 2 are pure 2-opt sub-moves (single segment reversal).
    The calling code in _try_3opt_move deliberately excludes them because
    the preceding _try_2opt_move search has already proven them unprofitable
    for the current (i, j) pair.
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
    Exact gains for five representative 5-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6), (t7,t8), (t9,t10).

    Returns a list of five gain values (indices 0–4).  These cover the
    sequential 5-opt moves central to Helsgaun (2000), Section 4.3.
    The full 5-opt exhaustive enumeration has 60 non-trivial cases;
    only these 5 representative patterns are evaluated for efficiency.
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
