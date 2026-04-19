"""
Shaw Removal (Relatedness) Operator Module.

This module implements the Shaw removal heuristic, which removes nodes that are
similar (related) to a seed node based on distance, time, and waste.

Also includes profit-based Shaw removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.shaw import shaw_removal
    >>> routes, removed = shaw_removal(routes, n_remove=5, dist_matrix=d, ...)
    >>> from logic.src.policies.helpers.operators.destroy.shaw import shaw_profit_removal
    >>> routes, removed = shaw_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def shaw_removal(  # noqa: C901
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Optional[Dict[int, float]] = None,
    time_windows: Optional[Dict[Any, Any]] = None,
    randomization_factor: float = 2.0,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Shaw Removal: Remove related customers based on multi-criteria similarity.

    Implements Algorithm 2 from Ropke & Pisinger (2006) exactly:
    1. Pick a random seed from the current solution and add it to D.
    2. Repeat until |D| = q:
       a. Pick a random node **r** from D (the already-removed set).
       b. Compute relatedness R(r, c) for every candidate c not in D.
       c. Sort candidates ascending by R (lower R = more related).
       d. Draw y ~ U(0,1) and select the node at index floor(y^p * |L|).
    3. Remove all nodes in D.

    Efficiency: O(n) selection using np.partition instead of O(n log n) sorting.

    Relatedness measure (Ropke & Pisinger 2006, Eq. 17):
        R(i,j) = φ * d(i,j)/d_max + χ * |tw_i - tw_j|/tw_max + ψ * |q_i - q_j|/q_max

    **VRPP Adaptation Note**: While the Ropke & Pisinger (2006) measure includes
    temporal components (T_ij), the VRPP-specific relatedness primarily relies
    on spatial distance and waste/profit difference.  Time windows are used when
    supplied via ``time_windows``.

    **Note on ω (omega) term**: The vehicle compatibility component (ω) is
    intentionally omitted.  Rationale: the target domain uses a homogeneous fleet.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        time_windows: Time window dict {node: (earliest, latest)} (optional).
        randomization_factor: Determinism parameter *p* (Ropke & Pisinger 2006).
            High p → near-deterministic selection of the most related node
            (index 0).  p = 1 → fully uniform random selection across all
            candidates.  Must satisfy p ≥ 1.
        phi: Distance weight in relatedness (φ).
        chi: Time window weight in relatedness (χ).
        psi: Waste/demand weight in relatedness (ψ).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed nodes.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    waste = wastes or {}
    time_windows = time_windows or {}

    # Build node map
    node_map = {}
    all_nodes = []
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_map[node] = (r_idx, pos)
            all_nodes.append(node)

    if not all_nodes:
        return routes, []

    if rng is None:
        rng = Random()

    # Pick random seed
    seed: int = rng.choice(all_nodes)
    removed: List[int] = [seed]

    # Normalize factors using only active route nodes (Ropke & Pisinger 2005)
    # This prevents geographic outliers from compressing all distance scores to near-zero
    # Calculate max distance only among nodes currently in routes
    active_distances = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                active_distances.append(dist_matrix[i, j])
    max_dist: float = float(max(active_distances)) if active_distances else 1.0

    max_waste: float = float(max(waste.values())) if waste else 1.0
    max_tw: float = 1.0
    if time_windows:
        tw_spans = [tw[1] - tw[0] for tw in time_windows.values() if tw]
        max_tw = float(max(tw_spans)) if tw_spans else 1.0

    while len(removed) < n_remove and len(removed) < len(all_nodes):
        # Algorithm 2, line 5: pick ONE random request from D as the pivot
        # (Ropke & Pisinger 2006).  This is the key structural distinction from
        # an average-relatedness approach: each iteration evaluates relatedness
        # to a *single* randomly chosen already-removed node, which preserves
        # the probabilistic chain of related request selection described in the
        # paper.
        pivot: int = rng.choice(removed)

        relatedness_scores: List[Tuple[int, float]] = []
        for node in all_nodes:
            if node in removed or node not in node_map:
                continue

            dist_rel = float(dist_matrix[node, pivot]) / max_dist if max_dist > 0 else 0.0
            waste_rel = 0.0
            if waste:
                waste_rel = float(abs(waste.get(node, 0.0) - waste.get(pivot, 0.0))) / max_waste
            tw_rel = 0.0
            if time_windows:
                tw_node = time_windows.get(node, (0.0, max_tw))
                tw_pivot = time_windows.get(pivot, (0.0, max_tw))
                tw_rel = float(abs(tw_node[0] - tw_pivot[0])) / max_tw

            rel = phi * dist_rel + chi * tw_rel + psi * waste_rel
            relatedness_scores.append((node, rel))

        if not relatedness_scores:
            break

        # Structured array for O(n) partitioning with deterministic tie-breaking (score, then node ID)
        dtype = [("node", "i4"), ("score", "f4")]
        arr = np.array(relatedness_scores, dtype=dtype)

        # Randomized selection using power law
        # y^p where y is uniform [0,1], p is randomization_factor
        y = rng.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        k = min(idx, len(relatedness_scores) - 1)

        # O(n) selection of the k-th smallest element
        # np.partition ensures elements at k are in their sorted position
        partitioned = np.partition(arr, k, order=["score", "node"])
        selected_node = int(partitioned[k]["node"])
        removed.append(selected_node)

    # Execution removals: efficient route modification using list filtering
    removed_set: Set[int] = set(removed)
    final_removed = []
    modified_routes = []
    for r in routes:
        new_route = [node for node in r if node not in removed_set]
        # Track removals from this specific route for final_removed
        for node in r:
            if node in removed_set:
                final_removed.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed


def shaw_profit_removal(  # noqa: C901
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    randomization_factor: float = 2.0,
    phi: float = 9.0,
    psi: float = 5.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Profit-based Shaw Removal (VRPP).

    Efficiency: O(n) selection using np.partition instead of O(n log n) sorting.

    Applies Algorithm 2 from Ropke & Pisinger (2006) adapted for profit-maximisation
    objectives.  Relatedness is measured purely by spatial distance and marginal
    profit difference, replacing the temporal component (T_ij) of the PDPTW
    formulation:

        R(i,j) = φ * d(i,j)/d_max + ψ * |profit_i - profit_j|/profit_range

    In each iteration a **single** random node is drawn from the already-removed
    set D (Algorithm 2, step 5) and used as the pivot for computing relatedness,
    exactly as specified in the paper.

    **Note on ω (omega) term**: The vehicle compatibility component (ω) is
    intentionally omitted.  Rationale: the target domain uses a homogeneous fleet.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        randomization_factor: Determinism parameter *p* (Ropke & Pisinger 2006).
            High p → near-deterministic selection of the most related node
            (index 0).  p = 1 → fully uniform random selection across all
            candidates.  Must satisfy p ≥ 1.
        phi: Distance weight in relatedness (φ).
        psi: Profit weight in relatedness (ψ).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Modified routes and removed nodes.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    # 1. Pre-calculate node profits using marginal formula
    node_map = {}
    all_nodes = []
    node_profits = {}
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_map[node] = (r_idx, pos)
            all_nodes.append(node)

            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            node_profits[node] = revenue - (detour_cost * C)

    if not all_nodes:
        return routes, []

    if rng is None:
        rng = Random()

    # 2. Pick random seed
    seed: int = rng.choice(all_nodes)
    removed: List[int] = [seed]

    # 3. Normalization factors using only active route nodes
    # Calculate max distance only among nodes currently in routes
    active_distances = []
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                active_distances.append(dist_matrix[i, j])
    max_dist = float(max(active_distances)) if active_distances else 1.0

    profit_vals = list(node_profits.values())
    profit_range = float(max(profit_vals) - min(profit_vals)) if len(profit_vals) > 1 else 1.0
    if profit_range == 0:
        profit_range = 1.0

    while len(removed) < n_remove and len(removed) < len(all_nodes):
        # Algorithm 2, line 5: pick ONE random request from D as pivot
        pivot: int = rng.choice(removed)

        relatedness_scores: List[Tuple[int, float]] = []
        for node in all_nodes:
            if node in removed or node not in node_map:
                continue

            dist_rel = float(dist_matrix[node, pivot]) / max_dist if max_dist > 0 else 0.0
            profit_rel = float(abs(node_profits[node] - node_profits[pivot])) / profit_range

            rel = phi * dist_rel + psi * profit_rel
            relatedness_scores.append((node, rel))

        if not relatedness_scores:
            break

        # Structured array for O(n) partitioning with deterministic tie-breaking (score, then node ID)
        dtype = [("node", "i4"), ("score", "f4")]
        arr = np.array(relatedness_scores, dtype=dtype)

        y = rng.random()
        idx = int((y**randomization_factor) * len(relatedness_scores))
        k = min(idx, len(relatedness_scores) - 1)

        # O(n) selection of the k-th smallest element
        partitioned = np.partition(arr, k, order=["score", "node"])
        selected_node = int(partitioned[k]["node"])
        removed.append(selected_node)

    # 4. Efficient route modification
    removed_set: Set[int] = set(removed)
    final_removed = []
    modified_routes = []
    for r in routes:
        new_route = [node for node in r if node not in removed_set]
        # Track removals from this specific route for final_removed
        for node in r:
            if node in removed_set:
                final_removed.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed
