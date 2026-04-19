"""
Branch-and-Bound (LDS) Insertion Repair Operator Module.

This module implements the re-insertion procedure described in Section 2.2 of:

    Shaw, P. (1998). Using Constraint Programming and Local Search Methods to
    Solve Vehicle Routing Problems. In Proceedings of the 4th International
    Conference on Principles and Practice of Constraint Programming (CP-98),
    Lectures Notes in Computer Science, Vol. 1520, Springer, pp. 417-431.

The operator re-inserts a set of removed visits into a partial plan via a
depth-first **Branch-and-Bound tree search** augmented with **Limited
Discrepancy Search** (LDS, Harvey & Ginsberg 1995).

LDS Logic:
    The first choice made by the heuristic (cheapest insertion) at each level
    costs 0 discrepancy. The k-th alternative costs (k-1) discrepancy units.
    The search terminates once the max discrepancy budget is reached. This
    effectively explores the "neighborhood" of the greedy heuristic path.

Algorithm (Section 2.2):
    Variable Selection  — Farthest Insertion (Section 2.2.1):
        Choose the unrouted visit v whose cheapest feasible insertion cost
        is highest ("farthest" in insertion-cost space), i.e., the visit that
        would be hardest to insert later. This front-loads the most
        constrained decisions.

    Value Selection — Cheapest First:
        Enumerate all feasible (route, position) pairs for v, sorted by
        increasing insertion cost delta. The first value is the heuristic
        choice (0 discrepancy consumed); each subsequent value costs one
        additional discrepancy unit.

    Pruning:
        If the partial plan cost (already-routed edges + a naive lower bound
        of zero for the unrouted visits) already equals or exceeds the best
        complete plan found so far, prune the branch.

    Termination:
        The tree terminates when all visits are placed (complete plan) or no
        discrepancy budget remains (leaf node) or pruned.

Two public function pairs are provided:

    * ``bb_insertion``        — standard, distance-minimising variant (CVRP).
    * ``bb_profit_insertion`` — profit-maximising VRPP variant with optional
                               speculative seeding of new routes.

Both share the internal recursive LDS engine ``_lds_reinsert``.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.branch_bound import (
    ...     bb_insertion, bb_profit_insertion
    ... )
    >>> routes = bb_insertion(routes, removed, dist_matrix, wastes, capacity)
    >>> routes = bb_profit_insertion(routes, removed, dist_matrix, wastes,
    ...                              capacity, R=1.0, C=1.0, max_discrepancy=3)
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)

# ---------------------------------------------------------------------------
# Shared internal types
# ---------------------------------------------------------------------------

# A "candidate position" for inserting node v: (cost_delta, route_idx, pos)
_Position = Tuple[float, int, int]

# Mutable plan state threaded through the LDS recursion
_Plan = List[List[int]]
_Loads = List[float]


# ---------------------------------------------------------------------------
# Internal helper: compute the total travel cost of all current routes
# ---------------------------------------------------------------------------


def _compute_plan_cost(routes: _Plan, dist_matrix: np.ndarray) -> float:
    """Compute the total routing cost (sum of edge lengths) across all routes.

    Each route is assumed to start and end at the depot (node 0).

    Args:
        routes: List of customer sequences per vehicle.
        dist_matrix: Square distance matrix (depot is index 0).

    Returns:
        Aggregate edge cost of all active routes.
    """
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += dist_matrix[0, route[0]]
        for k in range(len(route) - 1):
            total += dist_matrix[route[k], route[k + 1]]
        total += dist_matrix[route[-1], 0]
    return total


# ---------------------------------------------------------------------------
# Internal helper: enumerate + sort feasible insertion positions for one node
# ---------------------------------------------------------------------------


def _get_sorted_positions(
    node: int,
    routes: _Plan,
    loads: _Loads,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
) -> List[_Position]:
    """Return all feasible insertion positions for *node*, sorted by cost delta.

    A position is feasible when the receiving route does not exceed capacity
    after the insertion.  Each existing route and a (virtual) "new route"
    index ``len(routes)`` are evaluated.  The new-route option is always
    feasible and treated as a stand-alone out-and-back trip.

    Args:
        node: The customer node to be inserted.
        routes: Current partial plan (list of routes).
        loads: Cumulative load per route (parallel to *routes*).
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.

    Returns:
        List of ``(cost_delta, route_idx, position)`` triples, sorted
        ascending by *cost_delta* so that value ordering naturally produces
        cheapest-first branching.
    """
    node_waste = wastes.get(node, 0.0)
    positions: List[_Position] = []

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + node_waste > capacity:
            continue
        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
            positions.append((delta, r_idx, pos))

    # New route: out-and-back from depot  (always feasible when capacity allows)
    if node_waste <= capacity:
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        positions.append((new_cost, len(routes), 0))

    positions.sort(key=lambda x: x[0])
    return positions


# ---------------------------------------------------------------------------
# Internal helper: profit-aware feasible positions (VRPP)
# ---------------------------------------------------------------------------


def _get_sorted_positions_profit(
    node: int,
    routes: _Plan,
    loads: _Loads,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    is_mandatory: bool,
    seed_hurdle_factor: float = 0.5,
) -> List[Tuple[float, int, int]]:
    """Return feasible insertion positions for *node*, sorted by descending profit.

    For VRPP, positions are scored by ``profit = revenue - cost_delta * C``.
    Non-mandatory nodes at positions with negative profit are filtered out
    (unless no profitable option exists, in which case all are returned so
    that the caller can decide whether to include the node at all).

    Args:
        node: Customer node to be inserted.
        routes: Current partial plan.
        loads: Cumulative load per route.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance travelled.
        is_mandatory: If True, the node must be inserted even if unprofitable.
        seed_hurdle_factor: Minimum net profit expressed as a fraction of the
            new-route cost (default 0.5 → at least -50 % of new-route cost).
            Mirrors the speculative seeding hurdle used in other operators.

    Returns:
        List of ``(neg_profit, route_idx, position)`` triples sorted ascending
        by *neg_profit* (i.e. descending true profit) so that the LDS engine
        can treat cheapest-first and most-profitable-first uniformly using the
        same ascending-cost logic.
    """
    node_waste = wastes.get(node, 0.0)
    revenue = node_waste * R
    positions: List[Tuple[float, int, int]] = []

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + node_waste > capacity:
            continue
        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
            profit = revenue - delta * C
            if is_mandatory or profit >= -1e-4:
                # Store as -profit so ascending sort → highest profit first
                positions.append((-profit, r_idx, pos))

    # New route: speculative seeding
    if node_waste <= capacity:
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - new_cost * C
        seed_hurdle = -seed_hurdle_factor * new_cost * C
        if is_mandatory or new_profit >= seed_hurdle:
            positions.append((-new_profit, len(routes), 0))

    positions.sort(key=lambda x: x[0])
    return positions


# ---------------------------------------------------------------------------
# Core LDS engine — cost minimisation variant
# ---------------------------------------------------------------------------


def _lds_reinsert(  # noqa: C901
    routes: _Plan,
    loads: _Loads,
    unrouted: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    discrep: int,
    best_state: List[Optional[Tuple[float, _Plan, _Loads]]],
) -> None:
    """Recursive LDS tree search for cost-minimising re-insertion.

    Implements ``Reinsert(plan, visits, discrep)`` from Figure 2 of Shaw
    (1998).  The *best_state* list is used as a mutable singleton reference to
    propagate the globally optimal plan across recursive frames.

    Algorithm:
        1. **Base case:** if *unrouted* is empty, check whether the completed
           plan improves on the best known and update accordingly.
        2. **Variable selection:** apply Farthest Insertion ordering — select
           the visit *v* whose cheapest feasible insertion cost is the largest.
           This maximally constrains the search early and mimics Section 2.2.1.
        3. **Value ordering:** enumerate all feasible positions for *v*, sorted
           by increasing cost delta (cheapest first), matching the paper's
           cheapest-insertion value heuristic.
        4. **LDS branching:** the first position (the heuristic's first choice)
           costs 0 discrepancy.  Each subsequent position costs 1 discrepancy
           unit; a branch is abandoned once ``discrep < 0``.
        5. **Pruning:** before recursing, check if the current plan cost already
           exceeds the best known.  If the current objective already dominates
           the pruning bound, skip the subtree.

    Args:
        routes: Mutable list of routes representing the current plan state.
        loads: Mutable per-route accumulated load (parallel to *routes*).
        unrouted: Nodes that have not yet been assigned to any route.
        dist_matrix: Distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle capacity.
        discrep: Remaining discrepancy budget (non-negative).
        best_state: Single-element list ``[best]`` where each *best* is either
            ``None`` (no complete solution yet) or a
            ``(cost, routes_copy, loads_copy)`` tuple.
    """
    if not unrouted:
        # Leaf: evaluate complete plan
        cost = _compute_plan_cost(routes, dist_matrix)
        current_best = best_state[0]
        if current_best is None or cost < current_best[0]:
            # Deep copy to avoid mutation via backtracking
            best_state[0] = (cost, [list(r) for r in routes], list(loads))
        return

    # --- Farthest-Insertion variable ordering (Section 2.2.1) ---
    # Select v* = argmax_{v in unrouted} min_cost(v)
    # Use -math.inf (not -1.0) so that negative cost deltas (triangle-inequality
    # violations) are correctly handled.
    farthest_node = unrouted[0]  # safe fallback: always selects something
    farthest_min_cost = -math.inf

    for candidate in unrouted:
        positions = _get_sorted_positions(candidate, routes, loads, dist_matrix, wastes, capacity)
        min_cost = positions[0][0] if positions else dist_matrix[0, candidate] + dist_matrix[candidate, 0]
        if min_cost > farthest_min_cost:
            farthest_min_cost = min_cost
            farthest_node = candidate

    v = farthest_node
    remaining = [n for n in unrouted if n != v]

    # --- Value ordering: cheapest first ---
    positions = _get_sorted_positions(v, routes, loads, dist_matrix, wastes, capacity)

    # If no feasible position exists (e.g. all routes full), force a new route
    # (always feasible given one vehicle per removed visit as fallback)
    if not positions:
        new_cost = dist_matrix[0, v] + dist_matrix[v, 0]
        positions = [(new_cost, len(routes), 0)]

    # --- LDS branching ---
    # The i-th (0-indexed) branch consumes i discrepancy units
    current_best = best_state[0]
    current_plan_cost = _compute_plan_cost(routes, dist_matrix)

    for i, (delta, r_idx, pos) in enumerate(positions):
        remaining_discrep = discrep - i  # LDS: each non-first branch costs 1
        if remaining_discrep < 0:
            break  # Exhausted discrepancy budget

        # --- Pruning ---
        if current_best is not None:
            # Lower bound: current edges + delta (ignoring remaining unrouted)
            # This is the naive lower bound from Section 2.2
            lb = current_plan_cost + delta
            if lb >= current_best[0]:
                continue  # Prune; also prunes all subsequent (costlier) positions

        # --- Apply insertion ---
        node_waste = wastes.get(v, 0.0)
        new_route_opened = r_idx == len(routes)

        if new_route_opened:
            routes.append([v])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, v)
            loads[r_idx] += node_waste

        # --- Recurse ---
        _lds_reinsert(
            routes,
            loads,
            remaining,
            dist_matrix,
            wastes,
            capacity,
            remaining_discrep,
            best_state,
        )

        # --- Backtrack ---
        if new_route_opened:
            routes.pop()
            loads.pop()
        else:
            routes[r_idx].pop(pos)
            loads[r_idx] -= node_waste


# ---------------------------------------------------------------------------
# Core LDS engine — profit-maximisation variant (VRPP)
# ---------------------------------------------------------------------------


def _lds_reinsert_profit(  # noqa: C901
    routes: _Plan,
    loads: _Loads,
    unrouted: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_set: Set[int],
    discrep: int,
    best_state: List[Optional[Tuple[float, _Plan, _Loads]]],
    seed_hurdle_factor: float,
) -> None:
    """Recursive LDS tree search for profit-maximising re-insertion (VRPP).

    Analogous to ``_lds_reinsert`` but objective is *maximise profit* rather
    than *minimise distance*.  The best_state tuple carries
    ``(-profit, routes_copy, loads_copy)`` so the same ``<`` comparison works
    for maximisation (a larger negative value indicates a better plan).

    Non-mandatory nodes with no profitable position in any route or new route
    are silently dropped from the unrouted list.

    Args:
        routes: Mutable list of routes.
        loads: Mutable per-route accumulated load.
        unrouted: Nodes not yet assigned.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory_set: Node IDs that must be inserted regardless of profit.
        discrep: Remaining discrepancy budget.
        best_state: Single-element list containing ``(-profit, routes, loads)``
            or ``None``.
        seed_hurdle_factor: Fraction of new-route cost used as seed hurdle.
    """
    if not unrouted:
        # Compute total profit for the complete plan
        total_profit = 0.0
        for route in routes:
            if not route:
                continue
            # Route cost
            route_cost = dist_matrix[0, route[0]]
            for k in range(len(route) - 1):
                route_cost += dist_matrix[route[k], route[k + 1]]
            route_cost += dist_matrix[route[-1], 0]
            # Route revenue
            route_revenue = sum(wastes.get(n, 0.0) for n in route) * R
            total_profit += route_revenue - route_cost * C

        # Negate for minimisation-compatible comparison
        neg_profit = -total_profit
        current_best = best_state[0]
        if current_best is None or neg_profit < current_best[0]:
            best_state[0] = (neg_profit, [list(r) for r in routes], list(loads))
        return

    # --- Variable selection: Farthest Insertion (most constrained first) ---
    # Here "min_cost" is replaced by "max achievable profit" (choose node with
    # the smallest best profit → hardest to place profitably).
    farthest_node: Optional[int] = None
    farthest_best_profit = math.inf  # We want the *minimum* best profit

    droppable: List[int] = []  # Non-mandatory nodes with no feasible profitable positions

    for candidate in unrouted:
        is_mandatory = candidate in mandatory_set
        positions = _get_sorted_positions_profit(
            candidate, routes, loads, dist_matrix, wastes, capacity, R, C, is_mandatory, seed_hurdle_factor
        )
        if not positions:
            if not is_mandatory:
                droppable.append(candidate)
                continue
            # Mandatory node: force a new route as a fallback position
            new_cost = dist_matrix[0, candidate] + dist_matrix[candidate, 0]
            new_profit = wastes.get(candidate, 0.0) * R - new_cost * C
            best_profit_candidate = new_profit
        else:
            # Recall positions stored as (-profit, ...), so positions[0][0] is -best_profit
            best_profit_candidate = -positions[0][0]

        if best_profit_candidate < farthest_best_profit:
            farthest_best_profit = best_profit_candidate
            farthest_node = candidate

    # Remove permanently unprofitable optional nodes before recursing
    remaining_unrouted = [n for n in unrouted if n not in droppable]

    if farthest_node is None or farthest_node in droppable:
        # All remaining nodes are droppable; recurse with cleared list
        _lds_reinsert_profit(
            routes,
            loads,
            [],
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            mandatory_set,
            discrep,
            best_state,
            seed_hurdle_factor,
        )
        return

    v = farthest_node
    is_mandatory_v = v in mandatory_set
    remaining = [n for n in remaining_unrouted if n != v]

    positions = _get_sorted_positions_profit(
        v, routes, loads, dist_matrix, wastes, capacity, R, C, is_mandatory_v, seed_hurdle_factor
    )

    if not positions and is_mandatory_v:
        # Mandatory node: force a new route
        new_cost = dist_matrix[0, v] + dist_matrix[v, 0]
        positions = [(-((wastes.get(v, 0.0) * R) - new_cost * C), len(routes), 0)]

    # --- LDS branching (most-profitable first) ---
    current_best = best_state[0]

    for i, (neg_profit_delta, r_idx, pos) in enumerate(positions):
        remaining_discrep = discrep - i
        if remaining_discrep < 0:
            break

        # Pruning: upper bound on the maximum achievable profit from this branch
        # Current plan profit + best possible profit for v (a weak upper bound)
        # For VRPP, pruning by lower bound on neg_profit:
        # if we can establish current neg_profit + neg_profit_delta >= best neg_profit, prune
        if current_best is not None:
            # An upper bound on profit remaining: assume all other nodes achieve
            # their best possible profit (weak but cheap bound)
            lb_neg_profit = current_best[0]  # Best neg-profit so far (most negative = best plan)
            # If inserting v at this position yields neg_profit_delta > current_best[0],
            # and all future choices are at their best, this branch cannot improve.
            # Simplified bound: prune if this position's profit alone is worse than best plan
            if -neg_profit_delta <= lb_neg_profit and not is_mandatory_v:
                continue  # Prune

        node_waste = wastes.get(v, 0.0)
        new_route_opened = r_idx == len(routes)

        if new_route_opened:
            routes.append([v])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, v)
            loads[r_idx] += node_waste

        _lds_reinsert_profit(
            routes,
            loads,
            remaining,
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            mandatory_set,
            remaining_discrep,
            best_state,
            seed_hurdle_factor,
        )

        # Backtrack
        if new_route_opened:
            routes.pop()
            loads.pop()
        else:
            routes[r_idx].pop(pos)
            loads[r_idx] -= node_waste


# ---------------------------------------------------------------------------
# Public API — cost minimisation (CVRP)
# ---------------------------------------------------------------------------


def bb_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    max_discrepancy: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
) -> List[List[int]]:
    """Re-insert removed visits using Branch-and-Bound with Limited Discrepancy Search.

    Implements the ``Reinsert`` procedure from Section 2.2 of Shaw (1998).
    The operator explores the insertion search space via a depth-first tree
    search using:

    * **Farthest-Insertion** variable ordering (Section 2.2.1) — the most
      constrained visit (highest cheapest-insertion cost) is branched on first.
    * **Cheapest-first** value ordering — positions are tried in order of
      increasing insertion cost delta.
    * **LDS** discrepancy management — the i-th value choice at any node costs
      i discrepancy units; the budget is ``max_discrepancy``.
    * **Cost pruning** — branches where the accumulated cost cannot improve on
      the best complete plan found so far are abandoned.

    The best complete plan found within the LDS budget is returned.  If the
    budget is zero, only the greedy first-choice path is explored (equivalent
    to a deterministic cheapest insertion with farthest-first ordering).

    Args:
        routes: Partial plan; a list of customer sequences for each vehicle.
            The depot is implicitly node 0 and is NOT stored inside routes.
        removed_nodes: List of customer indices to be re-inserted.
        dist_matrix: Square distance matrix of shape ``(N+1, N+1)`` where
            index 0 is the depot.
        wastes: Mapping from node index to demand (waste volume).
        capacity: Maximum load per vehicle.
        max_discrepancy: LDS discrepancy budget *d*.  Larger values explore
            more alternatives at the cost of exponential tree growth.
            Shaw (1998) used *d = 1* for VRP; *d = 2* trades quality for
            speed.  Recommended: 1–3.
        mandatory_nodes: Optional list of node indices that must be inserted
            into the plan regardless of feasibility (each is given its own
            route as a last resort).
        expand_pool: If True, all currently unvisited nodes are treated as
            candidates, not only those in *removed_nodes*.  Set False for
            strict ALNS mode.

    Returns:
        List[List[int]]: Updated routes after inserting the removed visits.
            Empty routes are stripped from the result.

    Note:
        Implements Reinsert (Shaw 1998 §2.2):
        1. Uses Farthest-Insertion variable selection.
        2. Uses Limited Discrepancy Search (LDS) for tree management.
        3. Branches match the greedy path at zero discrepancy.

    Raises:
        ValueError: If *max_discrepancy* is negative.
    """
    if max_discrepancy < 0:
        raise ValueError(f"max_discrepancy must be non-negative, got {max_discrepancy}")

    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Build initial load vector
    loads: List[float] = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    # Determine candidate nodes
    visited: Set[int] = {n for r in routes for n in r}
    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unrouted: List[int] = sorted(set(range(1, n_nodes + 1)) - visited)
    else:
        unrouted = sorted(set(removed_nodes) - visited)
        if mandatory_nodes:
            unrouted = sorted(set(unrouted) | (mandatory_set - visited))

    if not unrouted:
        return routes

    # Run LDS tree search
    # best_state[0] = (cost, routes_copy, loads_copy) | None
    best_state: List[Optional[Tuple[float, _Plan, _Loads]]] = [None]

    _lds_reinsert(
        routes,
        loads,
        unrouted,
        dist_matrix,
        wastes,
        capacity,
        max_discrepancy,
        best_state,
    )

    # Apply best found plan (or fall back to current plan if search failed)
    if best_state[0] is not None:
        _, best_routes, _ = best_state[0]
        # Restore routes in-place (caller's list object)
        routes.clear()
        routes.extend([r for r in best_routes if r])
    else:
        # Fallback: this should only occur if all routes are full + no new route
        for orphan in unrouted:
            if orphan in mandatory_set:
                routes.append([orphan])

    return routes


# ---------------------------------------------------------------------------
# Public API — profit maximisation (VRPP)
# ---------------------------------------------------------------------------


def bb_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    max_discrepancy: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    seed_hurdle_factor: float = 0.5,
) -> List[List[int]]:
    """Re-insert removed visits via LDS tree search, maximising route profit.

    VRPP adaptation of ``bb_insertion``.  Extends the Shaw (1998) framework
    with profit-aware scoring (revenue - cost) and speculative seeding of new
    routes via a hurdle threshold.

    The search objective is to maximise the aggregate plan profit:

        profit = Σ_route [ R * Σ_v waste(v)  -  C * route_distance ]

    Non-mandatory nodes that have no profitable insertion option are silently
    dropped (consistent with the opt-in philosophy of VRPP).

    Args:
        routes: Partial plan (list of customer sequences, depot implicit).
        removed_nodes: Customer indices to be re-inserted.
        dist_matrix: Square distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance.
        max_discrepancy: LDS discrepancy budget.  Values of 1–3 are practical
            for ALNS repair calls.  Defaults to 2.
        mandatory_nodes: Nodes that must appear in the final plan.
        expand_pool: If True, all unvisited nodes are candidates (VRPP
            exploration mode).  Defaults to False (strict ALNS mode).
        seed_hurdle_factor: Fraction of new-route round-trip cost used as the
            minimum acceptable profit for opening a new route.  A value of 0.5
            means the node's profit must cover at least 50 % of its travel
            cost.  Defaults to 0.5 (matching other operators in the package).

    Returns:
        List[List[int]]: Updated routes after profit-maximising insertion.
            Economically unviable routes (negative profit, no mandatory nodes)
            are pruned from the final result.

    Raises:
        ValueError: If *max_discrepancy* is negative.
    """
    if max_discrepancy < 0:
        raise ValueError(f"max_discrepancy must be non-negative, got {max_discrepancy}")

    mandatory_set: Set[int] = set(mandatory_nodes) if mandatory_nodes else set()
    loads: List[float] = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    visited: Set[int] = {n for r in routes for n in r}
    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unrouted: List[int] = sorted(set(range(1, n_nodes + 1)) - visited)
    else:
        unrouted = sorted(set(removed_nodes) - visited)
        if mandatory_nodes:
            unrouted = sorted(set(unrouted) | (mandatory_set - visited))

    if not unrouted:
        return routes

    best_state: List[Optional[Tuple[float, _Plan, _Loads]]] = [None]

    _lds_reinsert_profit(
        routes,
        loads,
        unrouted,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_set,
        max_discrepancy,
        best_state,
        seed_hurdle_factor,
    )

    if best_state[0] is not None:
        _, best_routes, _ = best_state[0]
        routes.clear()
        routes.extend([r for r in best_routes if r])
    else:
        # Fallback: mandatory nodes get their own route
        for orphan in unrouted:
            if orphan in mandatory_set:
                routes.append([orphan])

    # Remove economically unviable routes (no mandatory content)
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
