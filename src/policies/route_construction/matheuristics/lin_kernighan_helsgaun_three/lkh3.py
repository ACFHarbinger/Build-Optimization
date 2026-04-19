"""
LKH-3 Heuristic Module.

Full implementation of the Lin-Kernighan-Helsgaun heuristic (LKH-3) for solving
the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP).

Key features (following Helsgaun 2000 and LKH-3 extension paper):

1. **Alpha-measure candidate sets** (Section 4.1, Helsgaun 2000):
   Edges pruned using sensitivity analysis on minimum spanning 1-trees. Only
   α-nearest neighbours are considered for inclusion in the tour.

2. **Subgradient optimisation / penalty vector** (Section 4.1):
   Node penalties π that transform C → D = cᵢⱼ + πᵢ + πⱼ, maximising the
   lower bound given by the minimum 1-tree length.

3. **Sequential k-opt moves (k = 2..5)** (Section 4.3):
   Exact gain computation for every reconnection case at each level, with
   early-termination via the positive-gain criterion.  Routines for higher-
   order moves (3 to 5-opt) are delegated to components in
   ``._tour_improvement``, with all final segment swaps executed via
   :func:`move_kopt_intra` from
   ``logic.src.policies.helpers.operators.intra_route.k_opt``.

4. **Candidate-set restricted search** (Section 3.2 / 4.1):
   Inner loops restricted to the α-nearest neighbours of each node, giving
   O(n · k_cand) search per node rather than O(n²).

5. **Don't-look bits** (Section 5.3):
   Nodes whose neighbourhood yielded no improvement are skipped until a
   neighbouring move changes their tour-adjacency.

6. **Double-bridge kick** (ILS perturbation):
   4-opt non-sequential perturbation that escapes local optima, delegated to
   :func:`double_bridge` from
   ``logic.src.policies.helpers.operators.perturbation.double_bridge``.

7. **Tour merging** (pool-based recombination):
   Combines shared edges from several elite tours to seed new searches.

8. **VRP penalty / lexicographic objective** (LKH-3):
   Objective = (penalty, cost); penalty = total excess demand. All
   improvement checks use ``is_better(p1, c1, p2, c2)``.

9. **_improve_tour** — outer driver.  Iterates over every non-masked node
   t1 and tries 2-opt first; if unsuccessful it attempts 3-, 4-, and 5-opt
   in order (each gated by an instance-size threshold).  Accepts the first
   improvement found (first-improvement strategy) and returns immediately.
   Uses *don't-look bits* (Helsgaun 2000, Section 5.3) to skip nodes whose
   neighbourhood was exhausted since the last move touched them.

References:
    Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
      traveling salesman heuristic. EJOR 126, 106-130.
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for constrained
      traveling salesman and vehicle routing problems.

Example:
    >>> tour, cost = solve_lkh3(distance_matrix)
    >>> tour, cost = solve_lkh3(distance_matrix, waste=demands, capacity=cap)
"""

from __future__ import annotations

import warnings
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.graph_augmentation import (
    augment_graph,
    decode_augmented_tour,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker import (
    build_load_state,
    update_load_state_after_move,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.objective import (
    compute_alpha_measures,
    get_candidate_set,
    get_score,
    is_better,
    solve_subgradient,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.popmusic import (
    popmusic_candidates,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
    merge_tours_best,
)
from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement import (
    _dynamic_kopt_search,
    _try_2opt_move,
    _try_3opt_move,
    _try_4opt_move,
    _try_5opt_move,
    _try_oropt_move,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

# ---------------------------------------------------------------------------
# Main improvement loop
# ---------------------------------------------------------------------------


def _improve_tour(  # noqa: C901
    curr_tour: List[int],
    curr_pen: float,
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    dont_look_bits: Optional[np.ndarray] = None,
    max_k_opt: int = 5,
    n_original: Optional[int] = None,
    dynamic_topology_discovery: bool = False,
) -> Tuple[List[int], float, float, bool, Optional[np.ndarray]]:
    """
    Execute one complete pass of sequential k-opt local search (k = 2..5).

    For each node t1 (not masked by a don't-look bit) and its successor t2
    in the current tour, the following hierarchy is attempted in order:

    1. **Dynamic Search** — if dynamic_topology_discovery is True, uses recursive B&B.
    2. **Exhaustive Hierarchy** — 2-opt, 3-opt, 4-opt, 5-opt patterns.
       move applied via _try_2opt_move with lazy lexicographic gate.
    2. **3-opt** — exact gain pre-screen for all seven patterns; move applied
       via :func:`move_kopt_intra` (k=3).  Restricted to n < 500.
    3. **4-opt** — exact gain pre-screen for three patterns; move applied via
       :func:`move_kopt_intra` (k=4).  Restricted to n < 300.
    4. **5-opt** — exact gain pre-screen for five patterns; move applied via
       :func:`move_kopt_intra` (k=5).  Restricted to n < 200.

    **Don't-look bit policy**: After an accepted k-opt move, only the nodes
    whose tour-adjacency physically changed have their don't-look bits reset.
    For 2-opt on (t1,t2,t3,t4): only t1 and t3 are new edge endpoints.

    Args:
        curr_tour: Current closed tour.
        curr_pen, curr_cost: Current (penalty, cost).
        candidates: α-nearest-neighbour candidate lists.
        distance_matrix: Cost matrix.
        waste, capacity: VRP parameters (None for TSP).
        rng: Random number generator forwarded to operators.
        dont_look_bits: Boolean array; nodes with True are skipped.
            Allocated up to max(nodes_count, n_original + n_vehicles) to
            avoid IndexError on augmented graphs.
        max_k_opt: Maximum k for k-opt moves (2-5).
        n_original: Original graph size before augmentation.

    Returns:
        (new_tour, new_penalty, new_cost, any_improvement, updated_bits).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Allocate dont_look_bits safely for augmented graphs
    if dont_look_bits is None:
        # Use max of nodes_count and distance_matrix size to handle
        # augmented dummy depots whose IDs may exceed nodes_count
        max_idx = max(nodes_count, len(distance_matrix))
        dont_look_bits = np.zeros(max_idx, dtype=bool)
    elif len(dont_look_bits) < nodes_count:
        # Resize if needed (e.g. after augmentation change)
        expanded = np.zeros(nodes_count, dtype=bool)
        expanded[: len(dont_look_bits)] = dont_look_bits
        dont_look_bits = expanded

    # Build load state for exact penalty queries (VRP mode only)
    load_state = None
    if waste is not None and capacity is not None and n_original is not None:
        load_state = build_load_state(curr_tour, waste, capacity, n_original)

    # Feature 1: Position lookup array (O(1) lookups instead of list.index)
    # Size to distance_matrix to handle augmented dummy depots safely.
    pos = np.full(len(distance_matrix), -1, dtype=np.int32)
    for idx, node in enumerate(curr_tour[:-1]):
        pos[node] = idx

    improved_overall = False
    for i in range(nodes_count):
        t1 = curr_tour[i]

        # Guard against IndexError on augmented dummy node IDs
        if t1 < len(dont_look_bits) and dont_look_bits[t1]:
            continue

        t2 = curr_tour[i + 1]
        found_improvement_for_t1 = False

        # ---- Dynamic Discovery (Recursive B&B) ----
        if dynamic_topology_discovery:
            res_dyn, rp_dyn, rc_dyn, improved_dyn = _dynamic_kopt_search(
                curr_tour,
                i,
                t1,
                t2,
                candidates,
                d,
                waste,
                capacity,
                rng,
                n_original=n_original,
                load_state=load_state,
                max_k=max_k_opt,
                pos=pos,
            )
            if improved_dyn and res_dyn is not None:
                curr_tour, curr_pen, curr_cost = res_dyn, rp_dyn, rc_dyn
                improved_overall = True
                found_improvement_for_t1 = True
                # Rebuild pos after accepted move
                for idx, node in enumerate(curr_tour[:-1]):
                    pos[node] = idx
                return curr_tour, curr_pen, curr_cost, True, dont_look_bits

            # If not improved in dynamic mode, set dont-look bit and skip exhaustive k-opt
            if t1 < len(dont_look_bits):
                dont_look_bits[t1] = True
            continue

        # ---- Or-opt (Segment Relocation 1, 2, 3 nodes) ----
        res_or, rp_or, rc_or, imp_or = _try_oropt_move(
            curr_tour,
            t1,
            i,
            candidates,
            d,
            waste,
            capacity,
            rng,
            n_original=n_original,
            load_state=load_state,
            pos=pos,
        )
        if imp_or and res_or is not None:
            curr_tour, curr_pen, curr_cost = res_or, rp_or, rc_or
            improved_overall = True
            found_improvement_for_t1 = True
            if t1 < len(dont_look_bits):
                dont_look_bits[t1] = False
            # Rebuild pos after accepted move
            for idx, node in enumerate(curr_tour[:-1]):
                pos[node] = idx
            return curr_tour, curr_pen, curr_cost, True, dont_look_bits

        # ---- Exhaustive 2-opt ----
        res_tour, res_p, res_c, res_imp, j = _try_2opt_move(
            curr_tour,
            i,
            t1,
            t2,
            candidates,
            d,
            waste,
            capacity,
            rng,
            n_original=n_original,
            load_state=load_state,
            pos=pos,
        )
        if res_imp and res_tour is not None:
            curr_tour, curr_pen, curr_cost = res_tour, res_p, res_c
            improved_overall = True
            found_improvement_for_t1 = True
            t3 = curr_tour[j] if j >= 0 else t1
            if t1 < len(dont_look_bits):
                dont_look_bits[t1] = False
            if t3 < len(dont_look_bits):
                dont_look_bits[t3] = False
            # Rebuild load state after accepted move
            if load_state is not None:
                load_state = update_load_state_after_move(
                    load_state,
                    curr_tour,
                    waste,  # type: ignore[arg-type]
                    capacity,  # type: ignore[arg-type]
                    n_original,  # type: ignore[arg-type]
                )
            # Rebuild pos after accepted move
            for idx, node in enumerate(curr_tour[:-1]):
                pos[node] = idx
            return curr_tour, curr_pen, curr_cost, True, dont_look_bits

        # ---- 3-opt ----
        if nodes_count < 500 and max_k_opt >= 3:
            for t3_cand in candidates[t2]:
                if t3_cand == t1 or (t2 < len(distance_matrix) and pos[t3_cand] < 0):
                    continue
                j = int(pos[t3_cand])
                if j <= i + 1:
                    continue
                t4 = curr_tour[j + 1]

                res_tour, res_p, res_c, res_imp = _try_3opt_move(
                    curr_tour,
                    i,
                    j,
                    t1,
                    t2,
                    t3_cand,
                    t4,
                    d,
                    waste,
                    capacity,
                    rng,
                    n_original=n_original,
                    load_state=load_state,
                    pos=pos,
                )
                if res_imp and res_tour is not None and is_better(res_p, res_c, curr_pen, curr_cost):
                    curr_tour, curr_pen, curr_cost = res_tour, res_p, res_c
                    improved_overall = True
                    found_improvement_for_t1 = True
                    # Reset only new edge endpoints
                    for nd in [t1, t3_cand]:
                        if nd < len(dont_look_bits):
                            dont_look_bits[nd] = False
                    if load_state is not None:
                        load_state = update_load_state_after_move(
                            load_state,
                            curr_tour,
                            waste,  # type: ignore[arg-type]
                            capacity,  # type: ignore[arg-type]
                            n_original,  # type: ignore[arg-type]
                        )
                    # Rebuild pos after accepted move
                    for idx, node in enumerate(curr_tour[:-1]):
                        pos[node] = idx
                    return curr_tour, curr_pen, curr_cost, True, dont_look_bits

                # ---- 4-opt ----
                if nodes_count < 300 and max_k_opt >= 4:
                    for k_idx in range(j + 2, nodes_count):
                        t5 = curr_tour[k_idx]
                        t6 = curr_tour[(k_idx + 1) % nodes_count]

                        res4, rp4, rc4, ri4 = _try_4opt_move(
                            curr_tour,
                            i,
                            j,
                            k_idx,
                            t1,
                            t2,
                            t3_cand,
                            t4,
                            t5,
                            t6,
                            d,
                            waste,
                            capacity,
                            rng,
                            n_original=n_original,
                            load_state=load_state,
                            pos=pos,
                        )
                        if ri4 and res4 is not None and is_better(rp4, rc4, curr_pen, curr_cost):
                            curr_tour, curr_pen, curr_cost = res4, rp4, rc4
                            improved_overall = True
                            found_improvement_for_t1 = True
                            for nd in [t1, t3_cand, t5]:
                                if nd < len(dont_look_bits):
                                    dont_look_bits[nd] = False
                            if load_state is not None:
                                load_state = update_load_state_after_move(
                                    load_state,
                                    curr_tour,
                                    waste,  # type: ignore[arg-type]
                                    capacity,  # type: ignore[arg-type]
                                    n_original,  # type: ignore[arg-type]
                                )
                            # Rebuild pos after accepted move
                            for idx, node in enumerate(curr_tour[:-1]):
                                pos[node] = idx
                            return curr_tour, curr_pen, curr_cost, True, dont_look_bits

                        # ---- 5-opt ----
                        if nodes_count < 200 and max_k_opt >= 5:
                            for l_idx in range(k_idx + 2, nodes_count):
                                t7 = curr_tour[l_idx]
                                t8 = curr_tour[(l_idx + 1) % nodes_count]
                                res5, rp5, rc5, ri5 = _try_5opt_move(
                                    curr_tour,
                                    i,
                                    j,
                                    k_idx,
                                    l_idx,
                                    t1,
                                    t2,
                                    t3_cand,
                                    t4,
                                    t5,
                                    t6,
                                    t7,
                                    t8,
                                    d,
                                    waste,
                                    capacity,
                                    rng,
                                    n_original=n_original,
                                    load_state=load_state,
                                    pos=pos,
                                )
                                if ri5 and res5 is not None and is_better(rp5, rc5, curr_pen, curr_cost):
                                    curr_tour, curr_pen, curr_cost = res5, rp5, rc5
                                    improved_overall = True
                                    found_improvement_for_t1 = True
                                    for nd in [t1, t3_cand, t5, t7]:
                                        if nd < len(dont_look_bits):
                                            dont_look_bits[nd] = False
                                    if load_state is not None:
                                        load_state = update_load_state_after_move(
                                            load_state,
                                            curr_tour,
                                            waste,  # type: ignore[arg-type]
                                            capacity,  # type: ignore[arg-type]
                                            n_original,  # type: ignore[arg-type]
                                        )
                                    # Rebuild pos after accepted move
                                    for idx, node in enumerate(curr_tour[:-1]):
                                        pos[node] = idx
                                    return curr_tour, curr_pen, curr_cost, True, dont_look_bits

        if not found_improvement_for_t1 and t1 < len(dont_look_bits):
            dont_look_bits[t1] = True
    return curr_tour, curr_pen, curr_cost, improved_overall, dont_look_bits


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------


def solve_lkh3(  # noqa: C901
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    waste: Optional[np.ndarray] = None,
    capacity: float = 100.0,
    revenue: float = 1.0,
    cost_unit: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
    coords: Optional[np.ndarray] = None,
    # LKH-3 parameters
    max_trials: int = 100,
    popmusic_subpath_size: int = 50,
    popmusic_trials: int = 50,
    popmusic_max_candidates: int = 5,
    max_k_opt: int = 5,
    use_ip_merging: bool = True,
    max_pool_size: int = 5,
    subgradient_iterations: int = 50,
    # LNS parameters
    profit_aware_operators: bool = False,
    alns_iterations: int = 10,
    plateau_limit: int = 10,
    deep_plateau_limit: int = 30,
    perturb_operator_weights: Optional[List[float]] = None,
    # Other parameters
    n_vehicles: int = 0,
    n_original: int = 0,
    candidate_set: Optional[Dict[int, List[int]]] = None,
    # System parameters
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    rng: Optional[Random] = None,
    seed: int = 42,
    dynamic_topology_discovery: bool = False,
    native_prize_collecting: bool = False,
) -> Tuple[List[List[int]], float, float]:
    """
    Solve a TSP or CVRP instance using the LKH-3 iterated local-search scheme.

    Algorithm outline (Helsgaun 2000 / LKH-3):

    1. **Initialisation** — nearest-neighbour or provided tour.
    2. **Candidate sets** — α-measure computed from minimum spanning tree.
    3. **Local-search loop** — repeated k-opt passes (k = 2..5) with
       don't-look bits until a local optimum is reached.  All k-opt moves
       are executed via :func:`move_kopt_intra`.
    4. **Perturbation** — double-bridge kick via :func:`double_bridge` to
       escape the basin of attraction of the current local optimum.
    5. **Tour-pool merging** — every 10 kicks, two elite tours from the pool
       are merged to create a new starting point.
    6. **Best-solution tracking** — lexicographic (penalty, cost) objective.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.
        initial_tour: Optional starting tour (closed).  Nearest-neighbour
            construction is used when not provided.
        waste: 1-D demand array for VRP.
        capacity: Vehicle capacity constraint.
        revenue: Revenue per unit collected (VRPP parameter).
        cost_unit: Cost per distance unit.
        mandatory_nodes: Nodes that must be visited.
        coords: Node coordinates (N×2) for geometric operators.
        max_trials: Maximum number of kicks / restarts.
        popmusic_subpath_size: POPMUSIC sub-path size.
        popmusic_trials: Number of POPMUSIC runs.
        popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
        max_k_opt: Maximum k for k-opt moves (2-5).
        use_ip_merging: If True, use IP-based tour recombination.
        max_pool_size: Maximum number of elite solutions to retain.
        profit_aware_operators: Toggle VRPP mode (subset selection).
        alns_iterations: Number of ALNS iterations (VRPP mode only).
        plateau_limit: Iterations before destroy-repair.
        deep_plateau_limit: Iterations before perturbation.
        perturb_operator_weights: Weights for perturbation operators.
        n_vehicles: Number of vehicles.
        n_original: Original graph size before augmentation (for penalty
            calculation with augmented dummy depots).  If None, equals ``n``.
        candidate_set: Pre-computed candidate neighbour lists.  When provided,
            the expensive MST/POPMUSIC candidate generation is skipped
            entirely, enabling warm-start across LNS iterations.
        recorder: Optional recorder for visualisation / diagnostics.
        np_rng: Numpy random generator; seeded from 42 if not provided.
        rng: Random number generator forwarded to operators.
        seed: Seed for the random number generator.

    Returns:
        (best_tour, best_cost, best_penalty) where best_tour is a closed node sequence.
    """
    n = len(distance_matrix)

    # Degeneracy check: zero off-diagonal distances cause infinite loops
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        if not np.all(distance_matrix[mask] > 0):
            warnings.warn(
                "Zero off-diagonal distances detected in the distance matrix. "
                "Co-located nodes may cause infinite loops in LKH-3 k-opt search.",
                RuntimeWarning,
                stacklevel=2,
            )

    if n < 3:
        tour = list(range(n)) + [0]
        cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(n))
        return [tour], float(cost), 0.0

    if np_rng is None:
        np_rng = np.random.default_rng(seed)

    # Bridge numpy RNG (used for array ops) → stdlib Random (operator interfaces)
    stdlib_rng = Random(seed) if rng is None else rng

    # --- Phase 3: Multi-vehicle Graph Augmentation ---
    # Dummy depots must be added BEFORE initialization or alpha-measure calculation.
    # We only augment if n_vehicles > 1 (single vehicle TSP/CVRP uses original graph).
    if n_vehicles > 1:
        # Convert waste array back to dict node_id: value for augment_graph
        waste_dict = {i: float(waste[i]) for i in range(1, len(waste))} if waste is not None else {}
        distance_matrix, waste, n_original = augment_graph(
            distance_matrix,
            waste_dict,
            n_vehicles=n_vehicles,
            capacity=capacity,
        )
    else:
        # Single vehicle or TSP
        if n_original <= 0:
            n_original = n

    # 1. Initialisation
    curr_tour = _initialize_tour(distance_matrix, initial_tour)

    # 2. Candidate sets — reuse cached if provided, else generate fresh
    if candidate_set is not None:
        candidates = candidate_set
    elif n > 1000:
        candidates = popmusic_candidates(
            distance_matrix,
            curr_tour,
            subpath_size=popmusic_subpath_size,
            n_runs=popmusic_trials,
            max_candidates=popmusic_max_candidates,
            np_rng=np_rng,
        )
    else:
        # Held-Karp subgradient optimization for pi-penalties
        pi = None
        if subgradient_iterations > 0:
            pi = solve_subgradient(distance_matrix, max_iterations=subgradient_iterations, n_original=n_original)

        alpha = compute_alpha_measures(distance_matrix, pi=pi)
        candidates = get_candidate_set(distance_matrix, alpha, max_candidates=popmusic_max_candidates)

    # --- Phase 3: Candidate Set Validation (Multi-vehicle) ---
    # If using a provided candidate set for an augmented graph, we must ensure:
    # 1. Dummy depots have entries (copied from main depot 0)
    # 2. Other nodes include dummy depots in their candidate lists if they include 0
    if candidates is not None and n_vehicles > 1:
        dummy_indices = list(range(n_original, len(distance_matrix)))
        depot_cands = candidates.get(0, [])
        new_candidates = candidates.copy()

        # 1. Add entries for dummies themselves
        for i in dummy_indices:
            if i not in new_candidates:
                new_candidates[i] = depot_cands[:]

        # 2. Reciprocal candidates: if node -> 0 is possible, node -> dummy is possible
        for node, cands in new_candidates.items():
            if 0 in cands:
                # Add dummy indices to candidates list (avoid duplicates)
                filtered_dummies = [d for d in dummy_indices if d not in cands]
                if filtered_dummies:
                    new_candidates[node] = cands + filtered_dummies

        candidates = new_candidates

    curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity, n_original)
    best_tour = curr_tour[:]
    best_pen, best_cost = curr_pen, curr_cost

    dont_look_bits: Optional[np.ndarray] = None

    # Tour pool for elite-solution recombination
    tour_pool: List[List[int]] = [best_tour[:]]

    # 3. Main iterated local-search loop
    for restart in range(max_trials):
        # Inner local-search until no improving k-opt move exists
        while True:
            curr_tour, curr_pen, curr_cost, improved_local, dont_look_bits = _improve_tour(
                curr_tour,
                curr_pen,
                curr_cost,
                candidates,
                distance_matrix,
                waste,
                capacity,
                stdlib_rng,
                dont_look_bits,
                max_k_opt,
                n_original,
                dynamic_topology_discovery=dynamic_topology_discovery,
            )
            if not improved_local:
                break

        # Update global best
        if is_better(curr_pen, curr_cost, best_pen, best_cost):
            best_tour = curr_tour[:]
            best_pen, best_cost = curr_pen, curr_cost
            tour_pool.append(best_tour[:])
            if len(tour_pool) > max_pool_size:
                tour_pool.pop(0)

        if recorder is not None:
            recorder.record(
                iteration=restart,
                best_obj=-best_cost,
                curr_cost=curr_cost,
                best_penalty=best_pen,
            )

        # Every 10 restarts: try tour-pool merging
        if restart > 0 and restart % 10 == 0 and len(tour_pool) >= 2:
            merged_tour = merge_tours_best(tour_pool, distance_matrix, use_ip=use_ip_merging)
            merged_pen, merged_cost = get_score(merged_tour, distance_matrix, waste, capacity, n_original)

            if is_better(merged_pen, merged_cost, best_pen, best_cost):
                best_tour = merged_tour[:]
                best_pen, best_cost = merged_pen, merged_cost
                tour_pool.append(merged_tour[:])
                if len(tour_pool) > max_pool_size:
                    tour_pool.pop(0)

            curr_tour = merged_tour
            curr_pen, curr_cost = merged_pen, merged_cost
            dont_look_bits = None
        else:
            # Double-bridge perturbation via the shared operator
            curr_tour = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity, n_original)
            dont_look_bits = None

    # Extract routes from tour
    routes = decode_augmented_tour(best_tour, n_original)
    return routes, best_cost, best_pen


# ---------------------------------------------------------------------------
# ALNS-Integrated Solver (Phase 2)
# ---------------------------------------------------------------------------


def solve_lkh3_with_alns(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    waste: Optional[np.ndarray] = None,
    capacity: float = 100.0,
    revenue: float = 1.0,
    cost_unit: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
    coords: Optional[np.ndarray] = None,
    # LKH-3 parameters
    max_trials: int = 100,
    popmusic_subpath_size: int = 50,
    popmusic_trials: int = 50,
    popmusic_max_candidates: int = 5,
    max_k_opt: int = 5,
    use_ip_merging: bool = True,
    max_pool_size: int = 5,
    subgradient_iterations: int = 50,
    # ALNS parameters
    profit_aware_operators: bool = False,
    alns_iterations: int = 100,
    plateau_limit: int = 10,
    deep_plateau_limit: int = 30,
    perturb_operator_weights: Optional[List[float]] = None,
    # Other parameters
    n_vehicles: int = 0,
    n_original: int = 0,
    candidate_set: Optional[Dict[int, List[int]]] = None,
    # System parameters
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    rng: Optional[Random] = None,
    seed: int = 42,
    dynamic_topology_discovery: bool = False,
    native_prize_collecting: bool = False,
) -> Tuple[List[List[int]], float, float]:
    """
    Solve VRP/VRPP using LKH-3 + Adaptive Large Neighborhood Search matheuristic.

    This function provides a unified interface that automatically dispatches to:
    - **VRPP mode** (``profit_aware_operators=True``): ALNS + LKH-3 for subset selection

    Algorithm (VRPP mode):
    1. Initialize with greedy profitable subset
    2. For each ALNS iteration:
       a. Optimize routes with LKH-3 k-opt
       b. If improved, update best and elite pool
       c. If plateau, apply destroy-repair operators
       d. If deep plateau, apply perturbation with elite
    3. Return best routes and objective

    Args:
        distance_matrix: (N×N) symmetric distance matrix.
        initial_tour: Optional starting tour (closed).  Nearest-neighbour
            construction is used when not provided.
        waste: 1-D demand array for VRP.
        capacity: Vehicle capacity constraint.
        revenue: Revenue per unit collected (VRPP parameter).
        cost_unit: Cost per distance unit.
        mandatory_nodes: Nodes that must be visited.
        coords: Node coordinates (N×2) for geometric operators.
        max_trials: Max LKH-3 trials per routing optimization.
        popmusic_subpath_size: POPMUSIC sub-path size.
        popmusic_trials: Number of POPMUSIC runs.
        popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
        max_k_opt: Maximum k for k-opt moves (2-5).
        use_ip_merging: If True, use IP-based tour recombination.
        max_pool_size: Maximum number of elite solutions to retain.
        profit_aware_operators: Toggle VRPP mode (subset selection).
        alns_iterations: Number of ALNS iterations (VRPP mode only).
        plateau_limit: Iterations before destroy-repair.
        deep_plateau_limit: Iterations before perturbation.
        perturb_operator_weights: Weights for perturbation operators.
        n_vehicles: Number of vehicles.
        n_original: Original graph size before augmentation (for penalty
            calculation with augmented dummy depots).  If None, equals ``n``.
        candidate_set: Pre-computed candidate neighbour lists.  When provided,
            the expensive MST/POPMUSIC candidate generation is skipped
            entirely, enabling warm-start across LNS iterations.
        recorder: Optional state recorder for visualization.
        np_rng: NumPy random generator.
        rng: Random generator.
        seed: Seed for the random number generator.

    Returns:
        Tuple of (routes, objective) where:
        - routes: List of routes (each is list of node IDs, no depot)
        - objective: Profit (VRPP) or negative cost (CVRP)

    Example:
        >>> # VRPP mode
        >>> routes, profit = solve_lkh3_with_alns(
        ...     distance_matrix=dist,
        ...     wastes={1: 10, 2: 20, 3: 30},
        ...     capacity=100,
        ...     revenue=2.0,
        ...     cost_unit=1.0,
        ...     profit_aware_operators=True,
        ...     mandatory_nodes=[1],
        ... )
    """
    # Build wastes dict from numpy array (skip depot at index 0)
    if waste is not None:
        wastes: Dict[int, float] = {i: float(waste[i]) for i in range(1, len(waste))}
    else:
        wastes = {}

    if np_rng is None:
        np_rng = np.random.default_rng(seed)

    if rng is None:
        rng = Random(seed)

    from logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search import (
        LKH3_ALNS,
    )

    solver = LKH3_ALNS(
        distance_matrix=distance_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue=revenue,
        cost_unit=cost_unit,
        profit_aware_operators=profit_aware_operators,
        mandatory_nodes=mandatory_nodes,
        coords=coords,
        np_rng=np_rng,
        rng=rng,
        seed=seed,
        recorder=recorder,
        max_pool_size=max_pool_size,
        n_original=n_original,
        R=revenue,
        C=cost_unit,
        perturb_operator_weights=perturb_operator_weights,
    )

    routes, objective, penalty = solver.solve(
        max_iterations=alns_iterations,
        lkh_trials=max_trials,
        n_vehicles=n_vehicles,
        plateau_limit=plateau_limit,
        deep_plateau_limit=deep_plateau_limit,
        popmusic_subpath_size=popmusic_subpath_size,
        popmusic_trials=popmusic_trials,
        popmusic_max_candidates=popmusic_max_candidates,
        max_k_opt=max_k_opt,
        use_ip_merging=use_ip_merging,
        subgradient_iterations=subgradient_iterations,
        dynamic_topology_discovery=dynamic_topology_discovery,
        native_prize_collecting=native_prize_collecting,
    )

    return routes, objective, penalty
