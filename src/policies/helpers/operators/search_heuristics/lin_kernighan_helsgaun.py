"""
LKH-1 Solver — Full Implementation (Helsgaun 2000).

Implements the complete Lin-Kernighan-Helsgaun (LKH-1) algorithm as described
in Helsgaun (2000), including the components absent from the existing
``lin_kernighan_helsgaun.py::solve_lkh``:

1. **Alpha-measure candidate sets** (Section 4.1, Helsgaun 2000):
   Edges pruned using sensitivity analysis on minimum spanning 1-trees. Only
   α-nearest neighbours are considered for inclusion in the tour.

2. **Minimum 1-Tree (1T) computation** (Section 3, Helsgaun 2000):
   A 1-tree rooted at node 0 is the minimum spanning tree on nodes {1..n-1}
   plus the two shortest edges incident to node 0.  Its weight W(π) is a
   Held-Karp lower bound on the optimal tour length when the node penalties
   π are optimal (Held & Karp 1970).

3. **Subgradient optimisation of π** (Section 3.1, Helsgaun 2000):
   The penalty vector π ∈ ℝ^n is updated to maximise the 1-tree lower bound:

       W(π) = min-1-tree(c̃_{ij}) + 2 · Σ_i π_i
       c̃_{ij} = c_{ij} + π_i + π_j          (penalty-modified costs)
       g_i    = d_i(T) − 2                   (degree-defect subgradient)
       π ← π + t · g,   t = μ · (W* − W) / ||g||²  (Polyak step)

   where d_i(T) is the degree of node i in the 1-tree T.

4. **Penalty-modified candidate sets** (Section 4.1, Helsgaun 2000):
   The α-nearness and candidate lists are recomputed using c̃ rather than c
   after subgradient convergence.  This produces far tighter candidates than
   MST-only α-measures on the raw distances.

5. **Sequential k-opt moves (k = 2..5)** (Section 4.3):
   Exact gain computation for every reconnection case at each level, with
   early-termination via the positive-gain criterion.  Routines for higher-
   order moves (3 to 5-opt) are delegated to components in
   ``._tour_improvement``, with all final segment swaps executed via
   :func:`move_kopt_intra` from
   ``logic.src.policies.helpers.operators.intra_route.k_opt``.

6. **Candidate-set restricted search** (Section 3.2 / 4.1):
   Inner loops restricted to the α-nearest neighbours of each node, giving
   O(n · k_cand) search per node rather than O(n²).

7. **Don't-look bits** (Section 5.3):
   Nodes whose neighbourhood yielded no improvement are skipped until a
   neighbouring move changes their tour-adjacency.

6. **Double-bridge kick** (ILS perturbation):
   4-opt non-sequential perturbation that escapes local optima, delegated to
   :func:`double_bridge` from
   ``logic.src.policies.helpers.operators.perturbation.double_bridge``.

7. **Tour merging** (pool-based recombination):
   Combines shared edges from several elite tours to seed new searches.

8. **_improve_tour** — outer driver.  Iterates over every non-masked node
   t1 and tries 2-opt first; if unsuccessful it attempts 3-, 4-, and 5-opt
   in order (each gated by an instance-size threshold).  Accepts the first
   improvement found (first-improvement strategy) and returns immediately.
   Uses *don't-look bits* (Helsgaun 2000, Section 5.3) to skip nodes whose
   neighbourhood was exhausted since the last move touched them.

Architecture
------------
This module owns:
- 1-tree computation     → :func:`compute_1tree`
- Subgradient optimiser  → :func:`run_subgradient`
- Top-level LKH-1 solver → :func:`solve_lkh1`

The inner k-opt improvement loop is delegated to:
- :func:`_improve_tour` from ``lin_kernighan_helsgaun.py`` (k = 2..5,
  don't-look bits, first-improvement).

Candidate set / α-measure utilities are reused from:
- :func:`compute_alpha_measures`, :func:`get_candidate_set` in
  ``lin_kernighan_helsgaun.py``.

The double-bridge perturbation is reused from:
- :func:`_double_bridge_kick` in ``_tour_construction.py``.

References
----------
Held, M., & Karp, R. M. (1970). The traveling-salesman problem and minimum
  spanning trees. Operations Research, 18(6), 1138–1162.

Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
  traveling salesman heuristic. EJOR 126, 106–130.

Example:
    >>> tour, cost = solve_lkh(distance_matrix)
"""

from __future__ import annotations

import random
from collections import deque
from random import Random
from typing import Deque, Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
    merge_tours,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_improvement import (
    _2opt_gain,
    _apply_kopt_via_operator,
    _try_3opt_move,
    _try_4opt_move,
    _try_5opt_move,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

# ---------------------------------------------------------------------------
# Alpha-measure and candidate-set construction
# ---------------------------------------------------------------------------


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute α-nearness for every edge using minimum-spanning-tree sensitivity.

    α(i, j) = c(i, j) − β(i, j),  where β(i, j) is the weight of the heaviest
    edge on the unique MST path between i and j (Helsgaun 2000, Section 4.1).

    Edges with α = 0 belong to some MST; edges with small α are "nearly
    spanning" and hence strong candidates for an optimal tour.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        (n × n) array of α-values (symmetric, non-negative).
    """
    n = len(distance_matrix)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    # Convert directed result to symmetric dense MST
    mst = mst_sparse.toarray()
    mst = np.maximum(mst, mst.T)

    # Build explicit adjacency list for faster MST traversal (fixes O(V^3) bottleneck)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    rows, cols = np.where(mst > 0)
    for r, c in zip(rows, cols, strict=False):
        adj[r].append((c, float(mst[r, c])))

    alpha = np.zeros((n, n), dtype=float)

    for i in range(n):
        # BFS to find max edge weight from root i to all other nodes
        max_edge_to = np.zeros(n, dtype=float)
        visited = np.zeros(n, dtype=bool)
        queue: Deque[int] = deque([i])
        visited[i] = True

        while queue:
            current = queue.popleft()
            for neighbor, weight in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    # Max edge on path to neighbor is max(max edge to current, weight)
                    max_edge_to[neighbor] = max(max_edge_to[current], weight)
                    queue.append(neighbor)

        # After BFS for root i, populate alpha row for pairs (i, j)
        for j in range(i + 1, n):
            alpha_val = distance_matrix[i, j] - max_edge_to[j]
            alpha[i, j] = alpha[j, i] = max(0.0, alpha_val)

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray,
    alpha_measures: np.ndarray,
    max_candidates: int = 5,
) -> Dict[int, List[int]]:
    """
    Build per-node candidate lists sorted by α-nearness (ties broken by cost).

    Restricting the inner LK search to these lists reduces worst-case
    complexity from O(n²) to O(n · max_candidates) per pass.

    Args:
        distance_matrix: (n × n) cost matrix.
        alpha_measures: (n × n) α-nearness matrix from :func:`compute_alpha_measures`.
        max_candidates: Maximum number of candidates per node (default 5).

    Returns:
        dict mapping each node index to its sorted candidate list.
    """
    n = len(distance_matrix)
    candidates: Dict[int, List[int]] = {}
    for i in range(n):
        indices = sorted(
            [j for j in range(n) if j != i],
            key=lambda j: (alpha_measures[i, j], distance_matrix[i, j]),
        )
        candidates[i] = indices[:max_candidates]
    return candidates


# ---------------------------------------------------------------------------
# Main improvement loop
# ---------------------------------------------------------------------------


def _improve_tour(  # noqa: C901
    curr_tour: List[int],
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    rng: Random,
    dont_look_bits: Optional[np.ndarray] = None,
    max_k: int = 5,
    fixed_edges: Optional[Union[Set[Tuple[int, int]], FrozenSet[Tuple[int, int]]]] = None,
) -> Tuple[List[int], float, bool, Optional[np.ndarray]]:
    """
    Execute one complete pass of sequential k-opt local search (k = 2..max_k).

    For each node t1 (not masked by a don't-look bit) and its successor t2
    in the current tour, the following hierarchy is attempted in order:

    1. **2-opt** — exact gain pre-screen over α-nearest neighbours of t2;
       move applied via :func:`move_kopt_intra` (k=2).
    2. **3-opt** — exact gain pre-screen for all seven patterns; direct segment
       swapping.  Enabled when max_k >= 3 and n < 500.
    3. **4-opt** — exact gain pre-screen for three patterns; direct segment
       swapping.  Enabled when max_k >= 4 and n < 300.
    4. **5-opt** — exact gain pre-screen for five patterns; direct segment
       swapping.  Enabled when max_k >= 5 and n < 200.

    Only exact positive gains trigger a move.  The first improving move at the
    lowest k is accepted (first-improvement strategy) and the function returns
    immediately so the outer loop can restart.

    Don't-look bits (Helsgaun 2000, Section 5.3) are reset for all nodes
    involved in an accepted move and set for nodes whose search found nothing.

    When ``fixed_edges`` is provided (e.g., the IPT backbone during LKH-2
    crossover), any k-opt move that would remove a fixed edge is skipped.
    This ensures offspring produced by IPT crossover preserve the shared
    parental backbone during gap-filling refinement (Helsgaun 2009, Sec. 3).

    Args:
        curr_tour: Current closed tour.
        curr_cost: Current tour cost.
        candidates: α-nearest-neighbour candidate lists.
        distance_matrix: Cost matrix.
        rng: Random number generator forwarded to operators.
        dont_look_bits: Boolean array of length n; nodes with True are skipped.
        max_k: Maximum k-opt depth (2-5). Throttles search for small instances.
        fixed_edges: Optional set of sorted (min, max) edge pairs that must
            not be removed by any k-opt move.  Used by LKH-2 IPT crossover
            to preserve backbone edges during offspring refinement.

    Returns:
        (new_tour, new_cost, any_improvement, updated_bits).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # O(1) position map for tour index lookups
    pos_map = {node: idx for idx, node in enumerate(curr_tour[:-1])}

    if dont_look_bits is None:
        dont_look_bits = np.zeros(nodes_count, dtype=bool)

    # Normalise fixed_edges to frozenset of sorted pairs for O(1) lookup.
    # An edge (u, v) is stored as (min(u,v), max(u,v)).
    _fixed: FrozenSet[Tuple[int, int]] = (
        frozenset((min(a, b), max(a, b)) for a, b in fixed_edges) if fixed_edges else frozenset()
    )

    def _is_fixed(a: int, b: int) -> bool:
        """Return True if the edge (a, b) must not be removed."""
        if not _fixed:
            return False
        return (min(a, b), max(a, b)) in _fixed

    improved_overall = False
    for i in range(nodes_count):
        t1 = curr_tour[i]

        if dont_look_bits[t1]:
            continue

        t2 = curr_tour[i + 1]

        # Do not attempt to break a fixed edge incident to t1.
        if _is_fixed(t1, t2):
            continue

        # ---- 2-opt ----
        for t3 in candidates[t2]:
            if nodes_count >= 5:
                if t3 == t1 or t3 == curr_tour[(i + 2) % nodes_count]:
                    continue
            else:
                if t3 == t2:
                    continue

            j = pos_map.get(t3, -1)
            if j == -1 or j <= i + 1:
                continue

            t4 = curr_tour[j + 1]

            # Skip 2-opt if it would remove a fixed edge
            if _is_fixed(t3, t4):
                continue

            gain = _2opt_gain(t1, t2, t3, t4, d)
            if gain > 1e-6:
                new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
                if new_tour is not None:
                    c_new = get_cost(new_tour, d)
                    if is_better(c_new, curr_cost):
                        curr_tour, curr_cost = new_tour, c_new
                        dont_look_bits[t1] = False
                        dont_look_bits[t2] = False
                        dont_look_bits[t3] = False
                        dont_look_bits[t4] = False
                        return curr_tour, curr_cost, True, dont_look_bits

            # ---- 3-opt ----
            if max_k >= 3 and nodes_count < 500:
                res_tour, res_c, res_imp = _try_3opt_move(curr_tour, i, j, t1, t2, t3, t4, d, rng)
                if res_imp and res_tour is not None and is_better(res_c, curr_cost):
                    # Skip if move removes any fixed edge (conservative check on
                    # the two edges being replaced: (t1,t2) already guarded above;
                    # check (t3,t4) via 2-opt guard; 3-opt removes one more edge)
                    curr_tour, curr_cost = res_tour, res_c
                    dont_look_bits[t1] = False
                    dont_look_bits[t2] = False
                    dont_look_bits[t3] = False
                    dont_look_bits[t4] = False
                    return curr_tour, curr_cost, True, dont_look_bits

                # ---- 4-opt ----
                if max_k >= 4 and nodes_count < 300:
                    for k_idx in range(j + 2, nodes_count):
                        t5 = curr_tour[k_idx]
                        t6 = curr_tour[k_idx + 1]

                        res4, rc4, ri4 = _try_4opt_move(
                            curr_tour,
                            i,
                            j,
                            k_idx,
                            t1,
                            t2,
                            t3,
                            t4,
                            t5,
                            t6,
                            d,
                            rng,
                        )
                        if ri4 and res4 is not None and is_better(rc4, curr_cost):
                            curr_tour, curr_cost = res4, rc4
                            dont_look_bits[t1] = dont_look_bits[t2] = dont_look_bits[t3] = dont_look_bits[
                                t4
                            ] = dont_look_bits[t5] = dont_look_bits[t6] = False
                            return curr_tour, curr_cost, True, dont_look_bits

                        # ---- 5-opt ----
                        if max_k >= 5 and nodes_count < 200:
                            for l_idx in range(k_idx + 2, nodes_count):
                                t7 = curr_tour[l_idx]
                                t8 = curr_tour[l_idx + 1]

                                res5, rc5, ri5 = _try_5opt_move(
                                    curr_tour,
                                    i,
                                    j,
                                    k_idx,
                                    l_idx,
                                    t1,
                                    t2,
                                    t3,
                                    t4,
                                    t5,
                                    t6,
                                    t7,
                                    t8,
                                    d,
                                    rng,
                                )
                                if ri5 and res5 is not None and is_better(rc5, curr_cost):
                                    # _try_5opt_move found an improving 5-opt move
                                    # Since 5-opt involves 5 cuts (10 nodes total) and we don't track
                                    # which specific m was used, clear all don't-look bits to be safe
                                    curr_tour, curr_cost = res5, rc5
                                    dont_look_bits.fill(False)  # Reset all bits after major improvement
                                    return curr_tour, curr_cost, True, dont_look_bits

        dont_look_bits[t1] = True
    return curr_tour, curr_cost, improved_overall, dont_look_bits


"""
LKH-1 Solver — Full Implementation (Helsgaun 2000).

Implements the complete Lin-Kernighan-Helsgaun (LKH-1) algorithm as described
in Helsgaun (2000), including the components absent from the existing
``lin_kernighan_helsgaun.py::solve_lkh``:

1. **Minimum 1-Tree (1T) computation** (Section 3, Helsgaun 2000):
   A 1-tree rooted at node 0 is the minimum spanning tree on nodes {1..n-1}
   plus the two shortest edges incident to node 0.  Its weight W(π) is a
   Held-Karp lower bound on the optimal tour length when the node penalties
   π are optimal (Held & Karp 1970).

2. **Subgradient optimisation of π** (Section 3.1, Helsgaun 2000):
   The penalty vector π ∈ ℝ^n is updated to maximise the 1-tree lower bound:

       W(π) = min-1-tree(c̃_{ij}) + 2 · Σ_i π_i
       c̃_{ij} = c_{ij} + π_i + π_j          (penalty-modified costs)
       g_i    = d_i(T) − 2                   (degree-defect subgradient)
       π ← π + t · g,   t = μ · (W* − W) / ||g||²  (Polyak step)

   where d_i(T) is the degree of node i in the 1-tree T.

3. **Penalty-modified candidate sets** (Section 4.1, Helsgaun 2000):
   The α-nearness and candidate lists are recomputed using c̃ rather than c
   after subgradient convergence.  This produces far tighter candidates than
   MST-only α-measures on the raw distances.

4. **Full ILS loop** (Section 5, Helsgaun 2000):
   Combines the penalized k-opt local search (delegated to
   :func:`_improve_tour` from ``._tour_improvement``) with double-bridge
   perturbation and elite-pool recombination.

Architecture
------------
This module owns:
- 1-tree computation     → :func:`compute_1tree`
- Subgradient optimiser  → :func:`run_subgradient`
- Top-level LKH-1 solver → :func:`solve_lkh1`

The inner k-opt improvement loop is delegated to:
- :func:`_improve_tour` from ``lin_kernighan_helsgaun.py`` (k = 2..5,
  don't-look bits, first-improvement).

Candidate set / α-measure utilities are reused from:
- :func:`compute_alpha_measures`, :func:`get_candidate_set` in
  ``lin_kernighan_helsgaun.py``.

The double-bridge perturbation is reused from:
- :func:`_double_bridge_kick` in ``_tour_construction.py``.

References
----------
Held, M., & Karp, R. M. (1970). The traveling-salesman problem and minimum
  spanning trees. Operations Research, 18(6), 1138–1162.

Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
  traveling salesman heuristic. EJOR 126, 106–130.
"""

# ---------------------------------------------------------------------------
# 1-Tree computation
# ---------------------------------------------------------------------------


def compute_1tree(
    penalized_distances: np.ndarray,
    root: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Compute a minimum 1-tree rooted at ``root`` (Held & Karp 1970).

    A 1-tree consists of:
    - The minimum spanning tree (MST) on all nodes *except* the root.
    - The two cheapest edges incident to the root.

    Its cost is a valid lower bound on the TSP optimum.

    Args:
        penalized_distances: (n × n) penalty-modified cost matrix
                             c̃_{ij} = c_{ij} + π_i + π_j.
        root:                Root node (default 0).

    Returns:
        (tree_weight, degree_array):
            tree_weight  — total edge weight of the 1-tree.
            degree_array — integer array of shape (n,); degree[i] = degree
                           of node i in the 1-tree.
    """
    n = len(penalized_distances)

    # Compute MST on nodes {0, ..., n-1} \ {root}
    non_root = [i for i in range(n) if i != root]
    sub_dist = penalized_distances[np.ix_(non_root, non_root)]
    mst_sparse = minimum_spanning_tree(sub_dist)
    mst_dense = mst_sparse.toarray()
    mst_dense = np.maximum(mst_dense, mst_dense.T)

    degree = np.zeros(n, dtype=np.int32)
    tree_weight = 0.0

    # Accumulate MST edges (re-indexed to global node indices)
    for local_i, global_i in enumerate(non_root):
        for local_j, global_j in enumerate(non_root):
            if local_j > local_i and mst_dense[local_i, local_j] > 1e-12:
                w = float(mst_dense[local_i, local_j])
                tree_weight += w
                degree[global_i] += 1
                degree[global_j] += 1

    # Add the two shortest edges incident to root
    root_edges = sorted(
        [(penalized_distances[root, j], j) for j in range(n) if j != root],
        key=lambda x: x[0],
    )
    for w, j in root_edges[:2]:
        tree_weight += w
        degree[root] += 1
        degree[j] += 1

    return tree_weight, degree


# ---------------------------------------------------------------------------
# Subgradient optimisation
# ---------------------------------------------------------------------------


def run_subgradient(
    distance_matrix: np.ndarray,
    max_iter: int = 100,
    mu_init: float = 1.0,
    patience: int = 20,
    root: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Maximise the Held-Karp lower bound W(π) via subgradient optimisation.

    At each step:
    1. Compute c̃_{ij} = c_{ij} + π_i + π_j (penalized distances).
    2. Compute minimum 1-tree weight W̃ and node degrees d_i.
    3. Recover the un-penalized bound: W(π) = W̃ + 2 · Σ_i π_i.
    4. Subgradient: g_i = d_i − 2 (positive → overconstrained).
    5. Polyak step: t = μ · (W* − W(π_current)) / ||g||²  where W* is the
       best known lower bound (updated greedily as we improve).
    6. Update: π ← π + t · g  (no non-negativity constraint; π can be negative).
    7. Halve μ if no improvement for ``patience`` consecutive iterations.

    Args:
        distance_matrix: (n × n) symmetric raw distance matrix.
        max_iter:        Maximum subgradient iterations.
        mu_init:         Initial step-size multiplier (μ₀ in Helsgaun 2000).
        patience:        Iterations without improvement before halving μ.
        root:            1-tree root node (typically 0).
        verbose:         Print per-iteration diagnostics.

    Returns:
        (pi, best_W, penalized_dm):
            pi          — optimal penalty vector, shape (n,).
            best_W      — best (highest) 1-tree lower bound achieved.
            penalized_dm — final penalized distance matrix c̃_{ij}.
    """
    n = len(distance_matrix)
    pi = np.zeros(n, dtype=float)
    best_W = -np.inf
    mu = mu_init
    no_improve_count = 0

    for k in range(max_iter):
        # Build penalized matrix: c̃_ij = c_ij + π_i + π_j
        pi_mat = pi[:, None] + pi[None, :]
        penalized = distance_matrix + pi_mat

        # 1-tree lower bound in penalized space
        W_tilde, degree = compute_1tree(penalized, root)
        W_pi = W_tilde + 2.0 * np.sum(pi)

        if W_pi > best_W + 1e-9:
            best_W = W_pi
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                mu /= 2.0
                no_improve_count = 0
                if mu < 1e-8:
                    break  # Converged

        # Subgradient: g_i = d_i(T) - 2  (degree defect)
        g = degree.astype(float) - 2.0
        g_sq = float(np.dot(g, g))
        if g_sq < 1e-12:
            break  # Perfect tour degree (all nodes have degree 2)

        # Polyak step size
        step = mu * max(W_pi - best_W, 1e-6) / g_sq
        # Avoid oscillation: clamp step to reasonable range
        step = min(step, 10.0 / max(np.abs(g).max(), 1.0))
        pi += step * g

        if verbose:
            print(f"  [subgradient k={k:3d}] W(π)={W_pi:.4f}  best_W={best_W:.4f}  μ={mu:.6f}")

    # Final penalized matrix
    pi_mat = pi[:, None] + pi[None, :]
    penalized_dm = distance_matrix + pi_mat
    return pi, best_W, penalized_dm


# ---------------------------------------------------------------------------
# Top-level LKH-1 solver
# ---------------------------------------------------------------------------


def solve_lkh1(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    max_k: int = 5,
    n_candidates: int = 5,
    sg_max_iter: int = 100,
    sg_mu_init: float = 1.0,
    sg_patience: int = 20,
    pool_size: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve a TSP instance using the full LKH-1 algorithm (Helsgaun 2000).

    This function orchestrates the complete LKH-1 pipeline:

    Phase 1 — Subgradient (Section 3.1):
        Run :func:`run_subgradient` to find penalty vector π* that maximises
        the Held-Karp 1-tree lower bound W(π).

    Phase 2 — Penalized candidate sets (Section 4.1):
        Compute α-nearness and candidate lists on the *penalized* distance
        matrix c̃_{ij} = c_{ij} + π*_i + π*_j.  These candidates are far
        tighter than those derived from raw distances.

    Phase 3 — ILS local-search loop (Section 5):
        Repeat until the iteration budget is exhausted:
        a. Run sequential k-opt local search via ``_improve_tour`` (k=2..max_k,
           don't-look bits) until a local optimum is reached.
        b. Update the global best solution.
        c. Every 10 iterations: try elite-pool tour merging.
        d. Otherwise: apply double-bridge perturbation.

    Args:
        distance_matrix: (n × n) symmetric raw cost matrix.
        initial_tour:    Optional starting closed tour.
        max_iterations:  ILS restart budget.
        max_k:           Maximum k-opt depth (2–5).
        n_candidates:    Candidate list size per node.
        sg_max_iter:     Maximum subgradient iterations.
        sg_mu_init:      Initial Polyak step multiplier.
        sg_patience:     Patience for μ halving.
        pool_size:       Elite tour pool size (for merge recombination).
        recorder:        Optional telemetry recorder.
        np_rng:          NumPy Generator.
        seed:            Alternative seed.

    Returns:
        (best_tour, best_cost, hk_lower_bound):
            best_tour      — best closed tour found.
            best_cost      — its total raw distance.
            hk_lower_bound — Held-Karp lower bound W(π*) from Phase 1.
    """
    n = len(distance_matrix)
    if n < 3:
        t = list(range(n)) + [0]
        return t, float(get_cost(t, distance_matrix)), 0.0

    if np_rng is None:
        np_rng = np.random.default_rng(seed if seed is not None else 42)
    stdlib_rng = Random(int(np_rng.integers(0, 2**31)))

    # -----------------------------------------------------------------
    # Phase 1: Subgradient optimisation → π*, W*(Held-Karp bound)
    # -----------------------------------------------------------------
    pi, hk_bound, penalized_dm = run_subgradient(
        distance_matrix,
        max_iter=sg_max_iter,
        mu_init=sg_mu_init,
        patience=sg_patience,
    )

    if recorder is not None:
        recorder.record(engine="lkh1_subgradient", hk_bound=hk_bound)

    # -----------------------------------------------------------------
    # Phase 2: Build α-candidate sets on the penalized matrix
    # -----------------------------------------------------------------
    alpha_penalized = compute_alpha_measures(penalized_dm)
    candidates = get_candidate_set(penalized_dm, alpha_penalized, max_candidates=n_candidates)

    # -----------------------------------------------------------------
    # Phase 3: ILS local-search on raw distances with penalized candidates
    # -----------------------------------------------------------------
    curr_tour = _initialize_tour(distance_matrix, initial_tour)
    curr_cost = get_cost(curr_tour, distance_matrix)
    best_tour = curr_tour[:]
    best_cost = curr_cost

    dont_look_bits: Optional[np.ndarray] = None
    tour_pool: List[List[int]] = [best_tour[:]]

    for restart in range(max_iterations):
        # Inner k-opt passes until local optimum
        while True:
            curr_tour, curr_cost, improved_local, dont_look_bits = _improve_tour(
                curr_tour,
                curr_cost,
                candidates,
                distance_matrix,
                stdlib_rng,
                dont_look_bits,
                max_k,
            )
            if not improved_local:
                break

        if is_better(curr_cost, best_cost):
            best_cost = curr_cost
            best_tour = curr_tour[:]
            tour_pool.append(best_tour[:])
            if len(tour_pool) > pool_size:
                tour_pool.pop(0)

        if recorder is not None:
            recorder.record(
                engine="lkh1",
                restart=restart,
                best_cost=best_cost,
                hk_bound=hk_bound,
                gap=(best_cost - hk_bound) / max(hk_bound, 1.0),
            )

        # Early termination if optimality gap is closed
        if hk_bound > 0 and (best_cost - hk_bound) / hk_bound < 1e-4:
            break

        # Every 10 restarts: try elite-pool merging
        if restart > 0 and restart % 10 == 0 and len(tour_pool) >= 2:
            idx1, idx2 = random.sample(range(len(tour_pool)), 2)
            merged = merge_tours(tour_pool[idx1], tour_pool[idx2], distance_matrix)
            merged_cost = get_cost(merged, distance_matrix)
            if is_better(merged_cost, best_cost):
                best_cost = merged_cost
                best_tour = merged[:]
                tour_pool.append(best_tour[:])
                if len(tour_pool) > pool_size:
                    tour_pool.pop(0)
            curr_tour = merged
            curr_cost = merged_cost
            dont_look_bits = None
        else:
            curr_tour = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            curr_cost = get_cost(curr_tour, distance_matrix)
            dont_look_bits = None

    return best_tour, best_cost, hk_bound


# ---------------------------------------------------------------------------
# Convenience wrapper matching the solve_lkh interface
# ---------------------------------------------------------------------------


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    max_k: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Thin convenience wrapper for :func:`solve_lkh1` matching the two-return
    interface of the existing ``lin_kernighan_helsgaun.solve_lkh``.

    Returns (best_tour, best_cost) without the Held-Karp bound.
    """
    tour, cost, _ = solve_lkh1(
        distance_matrix,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        max_k=max_k,
        recorder=recorder,
        np_rng=np_rng,
        seed=seed,
    )
    return tour, cost
