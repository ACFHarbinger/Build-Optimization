"""
Swap Star (SWAP*) Operator Module.

This module implements the SWAP* operator from Vidal (2022) for VRP.
SWAP* is an enhanced inter-route swap operator that removes two nodes from
different routes and reinserts each into the opposite route at the optimal position.

Reference:
    Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source
    implementation and SWAP* neighborhood. Computers & Operations Research, 140, 105643.

Implementation Notes:
    - This implementation achieves O(1) evaluation per Vidal (2022).
    - Uses Top-3 Insertion Cache: stores the 3 best insertion positions for each
      node into each route, enabling constant-time insertion cost evaluation.
    - Cache is maintained automatically by _update_map() after route modifications.
    - Complexity: O(1) per SWAP* evaluation (excluding cache updates which are
      amortized across all local search iterations).

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.inter_route.swap_star import move_swap_star
    >>> applied = move_swap_star(ls, u, v, r_u, p_u, r_v, p_v)

Removes u from route r_u and v from route r_v, then reinserts each node
into the other route at the best position. Only applies the move if it
improves the total cost by a threshold margin.

Args:
    ls: LocalSearch instance containing routes and distance matrix.
    u: Node to remove from route r_u.
    v: Node to remove from route r_v.
    r_u: Index of the route containing u.
    p_u: Position of u in route r_u.
    r_v: Index of the route containing v.
    p_v: Position of v in route r_v.

Returns:
    bool: True if the swap was applied (improving), False otherwise.
"""


def move_swap_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """SWAP* inter-route operator: swap nodes u and v between different routes.

    Uses O(1) evaluation via Top-3 Insertion Cache (Vidal 2022).
    Implements penalized search: capacity violations are allowed and penalized
    rather than strictly forbidden, enabling exploration of infeasible space
    to find shortcuts between feasible topologies (HGS paradigm).

    Removes u from route r_u and v from route r_v, then reinserts each node
    into the other route at the best position. Applies the move if it improves
    the total penalized cost (distance + capacity penalty).

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
            Must have penalty_capacity attribute for penalized search.
        u: Node to remove from route r_u.
        v: Node to remove from route r_v.
        r_u: Index of the route containing u.
        p_u: Position of u in route r_u.
        r_v: Index of the route containing v.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the swap was applied (improving penalized cost), False otherwise.
    """
    waste_u = ls.waste.get(u, 0)
    waste_v = ls.waste.get(v, 0)

    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]

    # Step 1: Calculate removal costs (negative = savings)
    prev_u = route_u[p_u - 1] if p_u > 0 else 0
    next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    gain_rem_u = ls.d[prev_u, u] + ls.d[u, next_u] - ls.d[prev_u, next_u]

    prev_v = route_v[p_v - 1] if p_v > 0 else 0
    next_v = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0
    gain_rem_v = ls.d[prev_v, v] + ls.d[v, next_v] - ls.d[prev_v, next_v]

    # Step 2: Calculate current and new route loads (HGS penalized search - no hard constraints)
    current_load_u = ls._get_load_cached(r_u)
    current_load_v = ls._get_load_cached(r_v)

    # New loads after swap
    new_load_u = current_load_u - waste_u + waste_v
    new_load_v = current_load_v - waste_v + waste_u

    # Calculate penalty delta (Vidal 2022: explore infeasible space with dynamic penalties)
    penalty_weight = getattr(ls, "penalty_capacity", 1.0)

    old_pen_u = penalty_weight * max(0.0, current_load_u - ls.Q)
    old_pen_v = penalty_weight * max(0.0, current_load_v - ls.Q)

    new_pen_u = penalty_weight * max(0.0, new_load_u - ls.Q)
    new_pen_v = penalty_weight * max(0.0, new_load_v - ls.Q)

    delta_penalty = (new_pen_u - old_pen_u) + (new_pen_v - old_pen_v)

    # For cache lookup, we still need temp routes
    temp_rv = route_v[:]
    temp_rv.pop(p_v)

    temp_ru = route_u[:]
    temp_ru.pop(p_u)

    # Step 3: O(1) evaluation of inserting u into route v (after v is removed)
    # Query Top-3 cache for u -> r_v
    best_delta_u = float("inf")
    best_pos_u = -1

    if u in ls.top_insertions and r_v in ls.top_insertions[u]:
        top_3_u = ls.top_insertions[u][r_v]
        # Find first valid insertion position (not relying on removed node v)
        for delta, pos in top_3_u:
            # Skip if original insertion position relied on node v
            # pos == p_v means inserting before v
            # pos == p_v + 1 means inserting after v
            if pos == p_v or pos == p_v + 1:
                continue

            best_delta_u = delta
            # Map to new array index (after v is removed)
            best_pos_u = pos if pos < p_v else pos - 1
            break

    # Fallback: check "in-place" insertion where v was
    cost_in_place_u = ls.d[prev_v, u] + ls.d[u, next_v] - ls.d[prev_v, next_v]
    if cost_in_place_u < best_delta_u:
        best_delta_u = cost_in_place_u
        best_pos_u = p_v

    # If no valid insertion found, reject
    if best_pos_u == -1:
        return False

    # Step 4: O(1) evaluation of inserting v into route u (after u is removed)
    # Query Top-3 cache for v -> r_u
    best_delta_v = float("inf")
    best_pos_v = -1

    if v in ls.top_insertions and r_u in ls.top_insertions[v]:
        top_3_v = ls.top_insertions[v][r_u]
        # Find first valid insertion position (not relying on removed node u)
        for delta, pos in top_3_v:
            # Skip if original insertion position relied on node u
            # pos == p_u means inserting before u
            # pos == p_u + 1 means inserting after u
            if pos == p_u or pos == p_u + 1:
                continue

            best_delta_v = delta
            # Map to new array index (after u is removed)
            best_pos_v = pos if pos < p_u else pos - 1
            break

    # Fallback: check "in-place" insertion where u was
    cost_in_place_v = ls.d[prev_u, v] + ls.d[v, next_u] - ls.d[prev_u, next_u]
    if cost_in_place_v < best_delta_v:
        best_delta_v = cost_in_place_v
        best_pos_v = p_u

    # If no valid insertion found, reject
    if best_pos_v == -1:
        return False

    # Step 5: Calculate total cost change including penalties (Vidal 2022: penalized search)
    # Distance delta (negative = improvement)
    total_delta = -gain_rem_u - gain_rem_v + best_delta_u + best_delta_v

    # Total cost change = distance change + penalty change
    total_cost_change = total_delta * ls.C + delta_penalty

    if total_cost_change < -1e-4:
        ls.routes[r_u] = temp_ru
        ls.routes[r_u].insert(best_pos_v, v)
        ls.routes[r_v] = temp_rv
        ls.routes[r_v].insert(best_pos_u, u)
        # _update_map automatically updates Top-3 cache for modified routes (Vidal 2022)
        ls._update_map({r_u, r_v})
        return True
    return False
