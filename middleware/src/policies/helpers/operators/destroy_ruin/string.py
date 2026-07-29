"""
String Removal Operator Module.

This module implements the string removal heuristic, which removes contiguous
sequences (strings) of nodes from routes to preserve local structure while
creating gaps for re-insertion.

Also includes profit-based string removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.string import string_removal
    >>> routes, removed = string_removal(routes, n_remove=5, dist_matrix=d)
    >>> from logic.src.policies.helpers.operators.destroy.string import string_profit_removal
    >>> routes, removed = string_profit_removal(routes, n_remove=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _sisr_ruin_pass(
    routes: List[List[int]],
    adjacency: List[Tuple[float, int]],
    ks: int,
    ls_max: float,
    rng: Random,
) -> Set[int]:
    """Iterate an adjacency list and remove one string per tour up to ks tours.

    Private helper shared by :func:`string_removal` and
    :func:`string_profit_removal`.

    Args:
        routes: Current routes (unmodified; referenced by index).
        adjacency: List of ``(score, customer)`` pairs sorted ascending by score.
            Customers earlier in the list are preferred anchors.
        ks: Number of distinct tours to ruin.
        ls_max: Maximum string length per tour (``ls_max`` from Eq. 8).
        rng: Random number generator.

    Returns:
        Set[int]: Set of node IDs selected for removal.
    """
    # Build a reverse mapping customer -> route_index from the *current* routes
    node_to_route: Dict[int, int] = {node: r_idx for r_idx, route in enumerate(routes) for node in route}

    ruined_tours: Set[int] = set()
    removed: Set[int] = set()

    for _, c in adjacency:
        if len(ruined_tours) >= ks:
            break
        if c in removed:
            continue
        r_idx = node_to_route.get(c, -1)
        if r_idx < 0 or r_idx in ruined_tours:
            continue

        route = routes[r_idx]
        if not route:
            continue

        try:
            anchor_pos: int = route.index(c)
        except ValueError:
            continue

        lt_max: int = max(1, min(len(route), int(ls_max)))
        lt: int = rng.randint(1, lt_max)

        start: int = anchor_pos
        end: int = min(start + lt, len(route))
        for node in route[start:end]:
            removed.add(node)

        ruined_tours.add(r_idx)

    return removed


def string_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    max_string_len: int = 4,
    avg_string_len: float = 3.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove contiguous strings of customers from adjacent tours (SISR).

    Faithfully implements Algorithm 2 of Christiaens & Vanden Berghe (2020),
    *Slack Induction by String Removals for Vehicle Routing Problems*:

    1. Derive global limits from ``avg_string_len`` (c̄) and ``max_string_len``
       (L_max):

       .. code-block:: text

           ls_max  = min(L_max, avg_tour_cardinality)          [Eq. 5]
           ks_max  = 4 * c̄ / (1 + ls_max) - 1                 [Eq. 6]
           ks      = floor(U(1, ks_max + 1))                   [Eq. 7]  # strings to remove

    2. Select a random seed customer ``c_seed`` from the solution.

    3. Build the **global adjacency list** ``adj(c_seed)``: all customers in the
       solution ordered by increasing Euclidean distance from ``c_seed`` (the
       seed itself is first).

    4. Iterate ``adj(c_seed)`` until ``ks`` distinct tours have been ruined:

       - If the customer ``c`` is still in the solution and its tour ``t`` has
         not yet been ruined:

         .. code-block:: text

             lt_max = min(|t|, ls_max)                         [Eq. 8]
             lt     = floor(U(1, lt_max + 1))                  [Eq. 9]
             Remove the string of length lt starting at c from tour t.
             Mark t as ruined.

    **Key constraints** (faithfully carried from the paper):
    - Each tour is ruined **at most once** (one string per tour).
    - The number of strings ``ks`` is drawn stochastically.
    - String lengths ``lt`` are drawn stochastically per tour.

    Args:
        routes: Current routes (list of node lists excluding the depot).
        n_remove: Target number of nodes to remove (used as ``c̄`` for
            ``ks_max`` when ``avg_string_len`` equals its default).
        dist_matrix: Distance/cost matrix ``(N+1, N+1)`` with depot at index 0.
        max_string_len: Maximum string cardinality ``L_max`` (default 4).
        avg_string_len: Average desired removals per call ``c̄``.  Used in
            ``Eq. 6`` to derive ``ks_max``.  Defaults to ``n_remove`` when
            the caller does not override it (backward-compatible sentinel -1
            resolves to ``n_remove`` internally).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    if rng is None:
        rng = Random()

    # Resolve c̄: treat default sentinel (3.0) as using n_remove directly so
    # the SISR formula is always meaningful.
    c_bar: float = float(n_remove) if avg_string_len <= 0 else avg_string_len

    # --- SISR Eq. 5 --------------------------------------------------------
    # Average tour cardinality (used to cap per-string length)
    non_empty = [r for r in routes if r]
    if not non_empty:
        return routes, []
    avg_tour_card: float = sum(len(r) for r in non_empty) / len(non_empty)
    ls_max: float = min(float(max_string_len), avg_tour_card)
    ls_max = max(ls_max, 1.0)

    # --- SISR Eq. 6 --------------------------------------------------------
    ks_max: float = max(1.0, (4.0 * c_bar) / (1.0 + ls_max) - 1.0)

    # --- SISR Eq. 7 --------------------------------------------------------
    # Draw ks ~ floor(U(1, ks_max + 1)) but clamp to available routes
    ks_draw: float = rng.uniform(1.0, ks_max + 1.0)
    ks: int = min(int(ks_draw), len(non_empty))
    ks = max(ks, 1)

    # --- Step 3: seed customer & global adjacency list -------------------
    all_customers: List[int] = [node for route in routes for node in route]
    if not all_customers:
        return routes, []
    c_seed: int = rng.choice(all_customers)

    # --- Build adjacency list from seed (sorted by distance) -----------
    adjacency: List[Tuple[float, int]] = [
        (float(dist_matrix[c_seed, c]) if c != c_seed else 0.0, c) for c in all_customers
    ]
    adjacency.sort(key=lambda x: (x[0], x[1]))

    # --- Ruin ks distinct tours via the adjacency list -------------------
    removed = _sisr_ruin_pass(routes, adjacency, ks, ls_max, rng)

    # --- Apply removals in a single pass ---------------------------------
    final_removed: List[int] = []
    modified_routes: List[List[int]] = []
    for r in routes:
        new_route: List[int] = []
        for node in r:
            if node in removed:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed


def string_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    max_string_len: int = 4,
    avg_string_len: float = 3.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove contiguous strings of customers, biased toward low-profit regions (VRPP).

    **VRPP Adaptation of SISR (Christiaens & Vanden Berghe 2020)**:
    Applies the same SISR Algorithm 2 structure (stochastic ``ks`` strings from
    distinct tours via adjacency list), with the following VRPP-specific
    modifications:

    - The **seed customer** is drawn from the bottom-25 % of nodes ranked by
      marginal profit (instead of uniformly at random), directing the destroy
      operator toward unprofitable regions of the solution.
    - The **adjacency list** from the seed is built using a combined score of
      distance and profit similarity (``score = d(seed, c) + alpha * |p_seed - p_c|``),
      so adjacent strings tend to share both spatial proximity and low-profit
      characteristics.

    String cardinality and the per-tour ruining logic follow Eqs. 5–9 of the
    paper exactly (see :func:`string_removal` for full formula details).

    Args:
        routes: Current routes (list of node lists excluding the depot).
        n_remove: Target number of nodes to remove.
        dist_matrix: Distance/cost matrix ``(N+1, N+1)`` with depot at index 0.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        max_string_len: Maximum string cardinality ``L_max`` (default 4).
        avg_string_len: Average desired removals per call ``c̄`` (default 3.0).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    if rng is None:
        rng = Random()

    # --- Pre-compute marginal profits ------------------------------------
    node_profits: Dict[int, float] = {}
    all_customers: List[int] = []

    for _r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            node_profits[node] = revenue - (detour_cost * C)
            all_customers.append(node)

    if not all_customers:
        return routes, []

    # --- SISR parameters (Eqs. 5-7) -------------------------------------
    c_bar: float = float(n_remove) if avg_string_len <= 0 else avg_string_len
    non_empty = [r for r in routes if r]
    avg_tour_card: float = sum(len(r) for r in non_empty) / len(non_empty)
    ls_max: float = max(1.0, min(float(max_string_len), avg_tour_card))
    ks_max: float = max(1.0, (4.0 * c_bar) / (1.0 + ls_max) - 1.0)
    ks_draw: float = rng.uniform(1.0, ks_max + 1.0)
    ks: int = min(int(ks_draw), len(non_empty))
    ks = max(ks, 1)

    # --- VRPP: low-profit seed selection --------------------------------
    sorted_by_profit = sorted(all_customers, key=lambda n: node_profits.get(n, 0.0))
    bottom_q = max(1, len(sorted_by_profit) // 4)
    c_seed: int = rng.choice(sorted_by_profit[:bottom_q])
    seed_profit: float = node_profits.get(c_seed, 0.0)

    # --- VRPP: profit-similarity adjacency list -------------------------
    # score = distance + 0.5 * |profit_seed - profit_c| (normalisation-free,
    # monotone in both proximity and profit similarity)
    adjacency: List[Tuple[float, int]] = []
    for c in all_customers:
        d = float(dist_matrix[c_seed, c]) if c != c_seed else 0.0
        profit_diff = abs(seed_profit - node_profits.get(c, 0.0))
        adjacency.append((d + 0.5 * profit_diff, c))
    adjacency.sort(key=lambda x: (x[0], x[1]))

    # --- Ruin ks distinct tours via adjacency list ----------------------
    removed = _sisr_ruin_pass(routes, adjacency, ks, ls_max, rng)

    # --- Apply removals --------------------------------------------------
    final_removed: List[int] = []
    modified_routes: List[List[int]] = []
    for r in routes:
        new_route: List[int] = []
        for node in r:
            if node in removed:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed
