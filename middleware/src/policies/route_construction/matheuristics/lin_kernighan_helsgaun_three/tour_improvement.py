"""
LKH-3 Tour Improvement Components.

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
   **CRITICAL FIX**: Now uses lazy evaluation — checks the lexicographic
   gate BEFORE constructing the candidate tour.

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
- :func:`get_score`, :func:`is_better` from ``._objective``
- :class:`TourAdapter` from ``._tour_adapter``
- :func:`_2opt_gain`, :func:`_3opt_gains`, :func:`_4opt_gains`,
  :func:`_5opt_gains` from ``._tour_construction``
- :func:`move_kopt_intra` from the shared intra-route operator package
- :func:`get_exact_penalty_delta` from ``load_tracker`` for O(L) penalty queries

Typical usage
-------------
>>> from logic.src.policies.lin_kernighan_helsgaun_three.tour_improvement import (
...     _apply_kopt_via_operator,
...     _try_2opt_move,
...     _try_3opt_move,
... )
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.kopt_topologies import (
    EXHAUSTIVE_2OPT_CASES,
    EXHAUSTIVE_3OPT_CASES,
    EXHAUSTIVE_4OPT_CASES,
    EXHAUSTIVE_5OPT_CASES,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker import (
    LoadState,
    get_exact_penalty_delta,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.objective import (
    get_score,
    is_better,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_construction import (
    _should_accept_kopt_move,
    build_tour_from_segments,
)

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
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """
    Search for an improving 2-opt move starting from edge (t1, t2).

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_2OPT_CASES constant
    (1 valid topology) for architectural consistency with higher-order k-opt.

    Algorithm (Lazy Evaluation Pattern):
    1. For each candidate t3 of t2:
       a. Extract broken nodes [t1, t2, t3, t4]
       b. Compute distance gain ΔC in O(1) using the single cached topology
       c. Compute penalty delta ΔP via load_state — O(L)
       d. Lexicographic gate: accept if ΔP<0 OR (ΔP≈0 AND ΔC>0)
       e. ONLY IF gate passes: construct tour and verify

    Args:
        curr_tour: Current closed tour.
        i: Position of t1 in the open route.
        t1, t2: Endpoints of the edge being considered for removal.
        candidates: α-nearest-neighbour lists.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters (None for TSP).
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved, j) where j is the position of t3.
        Returns (None, 0, 0, False, -1) if no improvement found.

    Complexity:
        - Per candidate: O(1) distance lookup + O(L) penalty check
        - Tour construction: O(N) only when gate passes
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Get current tour score for final verification
    p_curr, c_curr = get_score(curr_tour, d, waste, capacity, n_original)

    # Extract the single cached 2-opt topology (for architectural consistency)
    # EXHAUSTIVE_2OPT_CASES = [[(0, 2), (1, 3)]]
    assert len(EXHAUSTIVE_2OPT_CASES) == 1, "2-opt must have exactly 1 topology"
    topology_2opt = EXHAUSTIVE_2OPT_CASES[0]

    for t3_cand in candidates[t2]:
        if t3_cand == t1 or t3_cand == curr_tour[(i + 2) % nodes_count]:
            continue

        if pos is not None:
            if t3_cand >= len(pos) or pos[t3_cand] < 0:
                continue
            j = int(pos[t3_cand])
        else:
            try:
                j = curr_tour.index(t3_cand)
            except ValueError:
                continue

        if j <= i + 1:
            continue

        t4 = curr_tour[(j + 1) % nodes_count]

        # The 4 endpoints of the 2 broken edges (indexed 0..3 in cached topology)
        broken_nodes = [t1, t2, t3_cand, t4]
        broken_edges = [(t1, t2), (t3_cand, t4)]

        # Phase 1: Compute distance gain using cached topology
        # topology_2opt = [(0, 2), (1, 3)] maps to [(t1, t3), (t2, t4)]
        added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology_2opt]
        base_cost = d[t1, t2] + d[t3_cand, t4]
        added_cost = sum(d[u, v] for u, v in added_edges)
        delta_c = base_cost - added_cost

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                added_edges,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0
            if delta_c <= 1e-6:
                continue

        # Phase 3: Lexicographic gate
        if not _should_accept_kopt_move(delta_p, delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        new_tour = build_tour_from_segments(
            curr_tour,
            [i, i + 1, j, j + 1 if j + 1 < nodes_count else 0],
            topology_2opt,
            k=2,
        )

        # Phase 5: Full verification
        p2, c2 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p2, c2, p_curr, c_curr):
            return new_tour, p2, c2, True, j

    return None, 0.0, 0.0, False, -1


def _try_3opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 3-opt move with exhaustive topology enumeration.

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_3OPT_CASES constant
    (7 valid topologies) with lazy evaluation pattern.

    Algorithm (Lazy Evaluation Pattern):
    1. For each third cut position k_pos:
       a. Extract the 6 broken nodes [t1, t2, t3, t4, t5, t6]
       b. Iterate over ALL 7 cached 3-opt topologies
       c. Map cached indices [0..5] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 7 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Note: The 7 cached cases include 2 pure 2-opt sub-moves. These are
    evaluated anyway since the constant-factor overhead is negligible and
    they may find improving moves that were missed due to α-nearest neighbor
    filtering in the preceding _try_2opt_move search.

    Args:
        curr_tour: Current closed tour.
        i, j: Break-point positions of the first two cuts in the open route.
        t1..t4: Node pairs for the two existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improvement found.

    Complexity:
        - Per third-cut position: O(7) = O(1) topology evaluations
        - Distance lookups: O(7 × 3) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Iterate over all valid third cut positions
    for k_pos in range(j + 2, nodes_count):
        t5 = curr_tour[k_pos]
        t6 = curr_tour[(k_pos + 1) % nodes_count]

        # The 6 endpoints of the 3 broken edges (indexed 0..5 in cached topologies)
        broken_nodes = [t1, t2, t3, t4, t5, t6]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6]

        # Phase 1: Find best topology via O(1) distance lookups
        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        # Exhaustive enumeration over all 7 cached 3-opt topologies
        for topology in EXHAUSTIVE_3OPT_CASES:
            # Map cached indices [0..5] to actual node IDs
            # topology = [(u_idx, v_idx), ...] where each idx is in [0..5]
            # broken_nodes[idx] gives the actual node ID
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]

            # O(1) distance computation: sum of 3 edge costs
            added_cost = sum(d[u, v] for u, v in added_edges)

            # Distance gain (positive = improvement)
            delta_c = base_cost - added_cost

            # Track best topology
            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        # Sanity check (should never trigger with 7 cached cases)
        if best_topology is None:
            continue

        # Quick skip for TSP (no capacity constraints)
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        # Only evaluate capacity delta for the BEST topology found above
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate
        # Accept if: (1) reduces penalty OR (2) maintains feasibility AND improves cost
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        target_pos = [i, i + 1, j, j + 1, k_pos, k_pos + 1 if k_pos + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, target_pos, best_topology, k=3)

        # Phase 5: Full verification
        p3, c3 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p3, c3, curr_p, curr_c):
            return new_tour, p3, c3, True

    return None, 0.0, 0.0, False


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
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 4-opt move with exhaustive topology enumeration.

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_4OPT_CASES constant
    (25 valid non-sequential topologies) with lazy evaluation pattern.

    Algorithm (Lazy Evaluation Pattern):
    1. For each fourth cut position l:
       a. Extract the 8 broken nodes [t1, t2, t3, t4, t5, t6, t7, t8]
       b. Iterate over ALL 25 cached 4-opt topologies
       c. Map cached indices [0..7] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 25 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Args:
        curr_tour: Current closed tour.
        i, j, k: Break-point positions of the first three cuts.
        t1..t6: Node pairs for the three existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improvement found.

    Complexity:
        - Per fourth-cut position: O(25) = O(1) topology evaluations
        - Distance lookups: O(25 × 4) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Iterate over all valid fourth cut positions
    for l in range(k + 2, nodes_count):
        t7 = curr_tour[l]
        t8 = curr_tour[(l + 1) % nodes_count]

        # The 8 endpoints of the 4 broken edges (indexed 0..7 in cached topologies)
        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8]

        # Phase 1: Find best topology via O(1) distance lookups
        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        # Exhaustive enumeration over all 25 cached 4-opt topologies
        for topology in EXHAUSTIVE_4OPT_CASES:
            # Map cached indices [0..7] to actual node IDs
            # topology = [(u_idx, v_idx), ...] where each idx is in [0..7]
            # broken_nodes[idx] gives the actual node ID
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]

            # O(1) distance computation: sum of 4 edge costs
            added_cost = sum(d[u, v] for u, v in added_edges)

            # Distance gain (positive = improvement)
            delta_c = base_cost - added_cost

            # Track best topology
            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        # Sanity check (should never trigger with 25 cached cases)
        if best_topology is None:
            continue

        # Quick skip for TSP (no capacity constraints)
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        # Only evaluate capacity delta for the BEST topology found above
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,  # type: ignore[arg-type]
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate
        # Accept if: (1) reduces penalty OR (2) maintains feasibility AND improves cost
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        pos_indices = [i, i + 1, j, j + 1, k, k + 1, l, l + 1 if l + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos_indices, best_topology, k=4)

        # Phase 5: Full verification
        p4, c4 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p4, c4, curr_p, curr_c):
            return new_tour, p4, c4, True

    return None, 0.0, 0.0, False


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
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 5-opt move with exhaustive topology enumeration.

    **CRITICAL UPGRADE**: This function now evaluates ALL 208 valid non-sequential
    5-opt reconnection topologies using the module-level cached constant
    EXHAUSTIVE_5OPT_CASES. The previous implementation evaluated only 5
    representative cases, which was mathematically incomplete for publication.

    Algorithm:
    1. For each fifth cut position m:
       a. Extract the 10 broken nodes [t1, t2, ..., t10]
       b. Iterate over ALL 208 cached 5-opt topologies
       c. Map cached indices [0..9] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 208 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Complexity:
        - Per fifth-cut position: O(208) = O(1) topology evaluations
        - Distance lookups: O(208 × 5) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
        - Overall: O(N^5) amortized (same asymptotic as before, more thorough)

    Args:
        curr_tour: Current closed tour.
        i, j, k, l: Break-point positions of the first four cuts.
        t1..t8: Node pairs for the four existing broken edges.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters.
        rng: Random number generator (unused, kept for interface compatibility).
        n_original: Original graph size (for augmented dummy depot mode).
        load_state: Optional LoadState for penalty calculation.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improving move found.

    References:
        Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
        traveling salesman heuristic. European Journal of Operational Research,
        126(1), 106-130.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Iterate over all valid fifth cut positions
    for m in range(l + 2, nodes_count):
        t9 = curr_tour[m]
        t10 = curr_tour[(m + 1) % nodes_count]

        # The 10 endpoints of the 5 broken edges (indexed 0..9 in cached topologies)
        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8), (t9, t10)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8] + d[t9, t10]

        # Phase 1: Find best topology via O(1) distance lookups
        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        # Exhaustive enumeration over all 208 cached 5-opt topologies
        for topology in EXHAUSTIVE_5OPT_CASES:
            # Map cached indices [0..9] to actual node IDs
            # topology = [(u_idx, v_idx), ...] where each idx is in [0..9]
            # broken_nodes[idx] gives the actual node ID
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]

            # O(1) distance computation: sum of 5 edge costs
            added_cost = sum(d[u, v] for u, v in added_edges)

            # Distance gain (positive = improvement)
            delta_c = base_cost - added_cost

            # Track best topology
            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        # Sanity check (should never trigger with 208 cached cases)
        if best_topology is None:
            continue

        # Quick skip for TSP (no capacity constraints)
        if load_state is None and best_delta_c <= 1e-6:
            continue

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        # Only evaluate capacity delta for the BEST topology found above
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate
        # Accept if: (1) reduces penalty OR (2) maintains feasibility AND improves cost
        if not _should_accept_kopt_move(delta_p, best_delta_c):
            continue

        # Phase 4: Construct tour array (O(N) complexity)
        # ONLY executed when lexicographic gate passes
        pos_indices = [i, i + 1, j, j + 1, k, k + 1, l, l + 1, m, m + 1 if m + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos_indices, best_topology, k=5)

        # Phase 5: Full verification
        p5, c5 = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p5, c5, curr_p, curr_c):
            return new_tour, p5, c5, True

    return None, 0.0, 0.0, False


def __or_opt_relocation(
    seg_len,
    t1,
    t_first,
    t_after,
    t_last,
    t_dest,
    t_dest_after,
    d,
    load_state,
    waste,
    capacity,
    curr_tour,
    n_original,
    i,
    nodes_count,
    curr_p,
    curr_c,
):
    orientations = [True, False] if seg_len > 1 else [True]
    for forward in orientations:
        # Broken: (t1, t_first), (t_last, t_after), (t_dest, t_dest_after)
        # Added (Forward): (t1, t_after), (t_dest, t_first), (t_last, t_dest_after)
        # Added (Reverse): (t1, t_after), (t_dest, t_last), (t_first, t_dest_after)

        broken_edges = [(t1, t_first), (t_last, t_after), (t_dest, t_dest_after)]
        if forward:
            added_edges = [(t1, t_after), (t_dest, t_first), (t_last, t_dest_after)]
        else:
            added_edges = [(t1, t_after), (t_dest, t_last), (t_first, t_dest_after)]

        delta_c = (d[t1, t_first] + d[t_last, t_after] + d[t_dest, t_dest_after]) - (
            d[t1, t_after] + d[t_dest, added_edges[1][1]] + d[added_edges[2][0], t_dest_after]
        )

        if delta_c <= 1e-6:
            continue

        delta_p = 0.0
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour, broken_edges, added_edges, load_state, waste, capacity, n_original
            )

        if not _should_accept_kopt_move(delta_p, delta_c):
            continue

        # Construct new tour
        # Segment nodes: [(i+1)%nodes_count ... (i+seg_len)%nodes_count]
        seg = []
        for idx_s in range(i + 1, i + seg_len + 1):
            seg.append(curr_tour[idx_s % nodes_count])

        if not forward:
            seg.reverse()

        # Build the tour by removing the segment and re-inserting it
        # 1. Remove segment from original tour
        tour_no_seg = []
        # Segment indices to skip
        skip_indices = set((idx_s % nodes_count) for idx_s in range(i + 1, i + seg_len + 1))
        for idx_n in range(nodes_count):
            if idx_n not in skip_indices:
                tour_no_seg.append(curr_tour[idx_n])

        # 2. Re-insert after t_dest (which is now in tour_no_seg)
        try:
            insert_pos = tour_no_seg.index(t_dest)
            new_tour_open = tour_no_seg[: insert_pos + 1] + seg + tour_no_seg[insert_pos + 1 :]
            new_tour = new_tour_open + [new_tour_open[0]]
        except ValueError:
            continue  # Should not happen

        p_new, c_new = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p_new, c_new, curr_p, curr_c):
            return new_tour, p_new, c_new, True


def _try_oropt_move(
    curr_tour: List[int],
    t1: int,
    i: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Or-opt move: Relocate a segment of 1, 2, or 3 nodes.
    Relocates the segment [tour[i+1]...tour[i+len]] to follow node t_dest.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Segment lengths for Or-opt (standard LKH-3 uses 1, 2, 3)
    for seg_len in [3, 2, 1]:
        if nodes_count < seg_len + 3:
            continue

        # Segment nodes: [t_first, ..., t_last]
        # Current edges to break: (t1, t_first) and (t_last, t_after)
        t_first = curr_tour[(i + 1) % nodes_count]
        t_last = curr_tour[(i + seg_len) % nodes_count]
        t_after = curr_tour[(i + seg_len + 1) % nodes_count]

        # Try inserting this segment after each possible destination t_dest
        # We use candidates of t_first and t_last to find promising t_dest
        potential_dest_nodes = set()
        for cand in candidates[t_first]:
            potential_dest_nodes.add(cand)
        for cand in candidates[t_last]:
            potential_dest_nodes.add(cand)

        for t_dest in potential_dest_nodes:
            if t_dest in curr_tour[i + 1 : i + seg_len + 1]:
                continue

            # Find position of t_dest
            if pos is not None:
                if t_dest >= len(pos) or pos[t_dest] < 0:
                    continue
                dest_idx = int(pos[t_dest])
            else:
                try:
                    dest_idx = curr_tour.index(t_dest)
                except ValueError:
                    continue

            t_dest_after = curr_tour[(dest_idx + 1) % nodes_count]
            if t_dest_after == t_first:
                continue

            # Or-opt relocation can happen in two orientations (if seg_len > 1)
            # 1. Forward: ... t_dest, t_first ... t_last, t_dest_after ...
            # 2. Reverse: ... t_dest, t_last ... t_first, t_dest_after ...
            __or_opt_relocation(
                seg_len,
                t1,
                t_first,
                t_after,
                t_last,
                t_dest,
                t_dest_after,
                d,
                load_state,
                waste,
                capacity,
                curr_tour,
                n_original,
                i,
                nodes_count,
                curr_p,
                curr_c,
            )
    return None, 0.0, 0.0, False


def _dynamic_kopt_search(
    curr_tour: List[int],
    i: int,
    t1: int,
    t2: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    max_k: int = 5,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Dynamic, recursive k-opt search following true sequential LKH-3 principles.

    Refined Logic:
    1. Positive-G Pruning: Prunes if G_partial - cost(added_edge) <= 0.
    2. Sequential branching: t_{2k+2} is the unique non-backtracking neighbor of t_{2k+1}.
    3. Efficiency: Uses the O(1) pos lookup array passed from the driver.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # pos array is required for O(1) lookups
    if pos is None:
        pos = np.full(len(distance_matrix), -1, dtype=np.int32)
        for idx, node in enumerate(curr_tour[:-1]):
            pos[node] = idx

    def backtrack(
        k: int,
        t_list: List[int],
        gain_c: float,
    ) -> Tuple[Optional[List[int]], float, float, bool]:
        """
        Recursive core of the sequential LK search.
        k: current level (from 1 to max_k)
        t_list: points [t1, t2, ..., t_{2k}]
        gain_c: cumulative distance gain
        """
        if k > max_k:
            return None, 0.0, 0.0, False

        t_prev = t_list[-1]  # t_{2k}

        # Select t_{2k+1} from candidate list of t_{2k}
        for t_next in candidates[t_prev]:
            if t_next in t_list:
                continue

            # Step 1: Add edge (t_{2k}, t_{2k+1})
            added_c = d[t_prev, t_next]

            # Step 2: Positive-G Criterion (Prune if extension cannot help)
            # True LKH prunes when gain_c - added_c <= 0
            G_after_add = gain_c - added_c
            if G_after_add <= 0:
                continue

            # Step 3: Select t_{2k+2} - must keep the move sequential and non-backtracking
            idx_next = int(pos[t_next])  # type: ignore[index]
            pred = curr_tour[(idx_next - 1) % nodes_count]
            succ = curr_tour[(idx_next + 1) % nodes_count]
            t_neighbor = succ if pred == t_prev else pred

            if t_neighbor in t_list:
                continue

            # Step 4: Break edge (t_{2k+1}, t_{2k+2})
            broken_c = d[t_next, t_neighbor]
            new_gain_c = G_after_add + broken_c

            # Step 5: Try closure with (t_{2k+2}, t1)
            final_added_c = d[t_neighbor, t1]
            final_delta_c = new_gain_c - final_added_c

            if final_delta_c > 1e-6:
                # Validate closure and check penalty/score
                t_all = t_list + [t_next, t_neighbor]
                new_t, p_new, c_new, improved = _verify_and_construct(
                    curr_tour, t_all, d, waste, capacity, n_original, curr_p, curr_c, pos
                )
                if improved:
                    return new_t, p_new, c_new, True

            # Step 6: Recurse to depth k+1
            res_tour, res_p, res_c, res_imp = backtrack(k + 1, t_list + [t_next, t_neighbor], new_gain_c)
            if res_imp:
                return res_tour, res_p, res_c, True

        return None, 0.0, 0.0, False

    # Start recursion from k=1: existing broken edge (t1, t2)
    return backtrack(1, [t1, t2], d[t1, t2])


def _verify_and_construct(
    curr_tour: List[int],
    t_list: List[int],
    d: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int],
    curr_p: float,
    curr_c: float,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """Helper to verify k-opt closure and build resulting tour."""
    k = len(t_list) // 2
    if k < 2:
        return None, 0.0, 0.0, False

    # Get positions of t_i in current tour
    try:
        if pos is None:
            return None, 0.0, 0.0, False
        _pos = pos
        pos_list = [int(_pos[node]) for node in t_list]
    except (KeyError, IndexError, TypeError):
        return None, 0.0, 0.0, False

    # Sequential k-opt added edges (in t_list indexing): (1,2), (3,4) ... (2k-1, 0)
    added_edges_indices = []
    for i in range(k):
        added_edges_indices.append((2 * i + 1, (2 * i + 2) % (2 * k)))

    try:
        new_tour = build_tour_from_segments(curr_tour, pos_list, added_edges_indices, k=k)
        p, c = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p, c, curr_p, curr_c):
            return new_tour, p, c, True
    except (ValueError, IndexError):
        # build_tour_from_segments may fail if t_list doesn't form a Hamiltonian cycle
        pass
    return None, 0.0, 0.0, False
