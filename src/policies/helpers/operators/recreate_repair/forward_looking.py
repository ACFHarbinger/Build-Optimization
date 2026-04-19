"""
Forward-Looking Insertion Operator for Multi-Period ALNS.

This repair operator re-inserts a node into the horizon chromosome by
evaluating a joint delta objective that accounts for:

1. **Day-t routing cost delta** — the standard cheapest-insertion detour
   cost on day ``t``.
2. **Expected inventory penalty delta** — the change in overflow/holding
   cost across days ``t+1 … t+H`` induced by the new visit schedule,
   computed using the ``ScenarioTree`` probability structure.

The forward-looking evaluation is the key novelty of Coelho et al. (2012)
style inventory-routing ALNS, and is what differentiates the multi-period
ALNS from a naive day-by-day greedy repair.

Mathematical formulation
------------------------
For each candidate insertion position ``(t, route r, position p)``::

    Δobj = Δrouting_cost_t + λ · E[ΔOverflow_{t+1..t+H}(y_new)]

where::

    Δrouting_cost_t = dist[prev, node] + dist[node, next] - dist[prev, next]

    E[ΔOverflow_{t+1..t+H}(y_new)] = Σ_{τ=t+1}^{t+H}
        Σ_ξ P(ξ) · max(0, I_{node,τ}(ξ, y_new) - B_cap)

and ``I_{node,τ}(ξ, y_new)`` propagates inventory under scenario ``ξ``
with the new visit schedule ``y_new``.

References
----------
Coelho, L. C., Cordeau, J.-F., & Laporte, G. (2012). "The inventory-routing
problem with transshipment." Computers & Operations Research, 39(11).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _compute_routing_delta(
    route: List[int],
    node: int,
    position: int,
    dist_matrix: Any,
    C: float,
) -> float:
    """Compute cheapest-insertion routing cost delta.

    Args:
        route: Existing route (without depot sentinels).
        node: Node to insert.
        position: Index in ``route`` where ``node`` is inserted before.
        dist_matrix: Square distance/cost matrix (numpy array or similar).
        C: Cost per unit distance.

    Returns:
        Routing cost increase (positive = more expensive).
    """
    n = len(route)
    prev = 0 if position == 0 else route[position - 1]
    nxt = 0 if position == n else route[position]

    delta = float(dist_matrix[prev][node]) + float(dist_matrix[node][nxt]) - float(dist_matrix[prev][nxt])
    return delta * C


def _simulate_inventory_forward(
    node: int,
    visit_days: List[int],
    initial_fill: float,
    demand_scenarios: List[List[float]],
    bin_capacity: float,
    t_start: int,
    H: int,
    stockout_penalty: float,
) -> float:
    """Compute expected forward overflow cost.

    Simulates the inventory of ``node`` from day ``t_start`` to
    ``t_start + H`` under all provided demand scenarios.  Each scenario
    is an equally-weighted demand sequence ``demand[day_in_scenario]``.

    Args:
        node: Node identifier (used as index offset if needed).
        visit_days: Days in ``[t_start, t_start+H]`` where the node IS visited.
        initial_fill: Fill level entering day ``t_start`` (percentage 0–100).
        demand_scenarios: List of demand sequences.  Each entry is a list of
            daily demand increments over the lookahead window.
        bin_capacity: Bin capacity in the same units as ``initial_fill``.
        t_start: First day of the lookahead window.
        H: Lookahead depth (number of days).
        stockout_penalty: Cost per unit of overflow.

    Returns:
        Expected overflow cost across all scenarios.
    """
    if not demand_scenarios:
        return 0.0

    visit_set = set(visit_days)
    total_cost = 0.0

    for scenario_demands in demand_scenarios:
        fill = initial_fill
        for step in range(H):
            # Accumulate demand
            demand_inc = scenario_demands[step] if step < len(scenario_demands) else 0.0
            fill = min(fill + demand_inc, bin_capacity * 2)  # allow temporary overflow
            # Collection: if visited, collect to 0
            if (t_start + step) in visit_set:
                fill = 0.0
            # Overflow penalty
            overflow = max(0.0, fill - bin_capacity)
            total_cost += overflow * stockout_penalty

    # Average over scenarios
    return total_cost / len(demand_scenarios)


def forward_looking_insertion(  # noqa: C901
    horizon_routes: List[List[List[int]]],
    removed: List[Tuple[int, int]],
    dist_matrix: Any,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    scenario_tree: Optional[Any] = None,
    stockout_penalty: float = 500.0,
    look_ahead_days: int = 3,
    lambda_inventory: float = 1.0,
) -> List[List[List[int]]]:
    """Re-insert removed (node, target_day) pairs with forward-looking evaluation.

    For each ``(node, target_day)`` pair in ``removed``, this operator finds
    the insertion position in ``horizon_routes[target_day]`` that minimises
    the combined objective:

        Δobj = Δrouting_cost_{target_day}
             + λ · E[ΔOverflow_{target_day+1 .. target_day+H}]

    The inventory lookahead uses the ``ScenarioTree`` if available, otherwise
    falls back to zero expected inventory delta (degrades to standard greedy).

    Args:
        horizon_routes: Full T-day solution ``[day][route][node]``.
        removed: List of ``(node_id, target_day)`` pairs to re-insert.
        dist_matrix: Square distance matrix.
        wastes: Mapping ``{node_id: fill_level}`` for capacity checks.
        capacity: Vehicle capacity per route.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        scenario_tree: Optional ``ScenarioTree`` for future demand sampling.
        stockout_penalty: Penalty per unit of overflow.
        look_ahead_days: Lookahead depth H.
        lambda_inventory: Weight on inventory penalty delta (``λ``).

    Returns:
        Updated ``horizon_routes`` with all removed nodes re-inserted.
    """
    T = len(horizon_routes)

    for node, target_day in removed:
        t = min(max(target_day, 0), T - 1)

        # --- Retrieve demand scenarios for lookahead ---
        demand_scenarios: List[List[float]] = []
        if scenario_tree is not None and hasattr(scenario_tree, "get_scenarios_at_day"):
            H = min(look_ahead_days, T - t - 1)
            for step in range(1, H + 1):
                day_scenarios = scenario_tree.get_scenarios_at_day(t + step)
                if not day_scenarios:
                    break
                if not demand_scenarios:
                    demand_scenarios = [[0.0] * H for _ in range(len(day_scenarios))]
                for s_idx, sc in enumerate(day_scenarios):
                    if s_idx < len(demand_scenarios) and hasattr(sc, "wastes"):
                        node_demand = float(sc.wastes[node - 1]) if node - 1 < len(sc.wastes) else 0.0
                        if step - 1 < len(demand_scenarios[s_idx]):
                            demand_scenarios[s_idx][step - 1] = node_demand
        else:
            H = 0

        # --- Current inventory state ---
        initial_fill = wastes.get(node, 0.0)
        bin_capacity = 100.0  # normalised percentage

        # --- Existing visit schedule for this node (for baseline cost) ---
        current_visit_days: List[int] = []
        for tau in range(T):
            for r in horizon_routes[tau]:
                if node in r:
                    current_visit_days.append(tau)
                    break

        # Baseline expected inventory cost (without new insertion)
        baseline_cost = (
            _simulate_inventory_forward(
                node=node,
                visit_days=current_visit_days,
                initial_fill=initial_fill,
                demand_scenarios=demand_scenarios,
                bin_capacity=bin_capacity,
                t_start=t + 1,
                H=H,
                stockout_penalty=stockout_penalty,
            )
            if demand_scenarios
            else 0.0
        )

        day_routes = horizon_routes[t]

        # --- Find best insertion position ---
        best_delta = float("inf")
        best_route_idx = -1
        best_pos = 0

        node_demand = wastes.get(node, 0.0)

        for r_idx, route in enumerate(day_routes):
            # Capacity check
            current_load = sum(wastes.get(n, 0.0) for n in route)
            if current_load + node_demand > capacity:
                continue

            for pos in range(len(route) + 1):
                routing_delta = _compute_routing_delta(route, node, pos, dist_matrix, C)

                # Forward inventory evaluation
                new_visit_days = sorted(set(current_visit_days) | {t})
                new_inventory_cost = (
                    _simulate_inventory_forward(
                        node=node,
                        visit_days=new_visit_days,
                        initial_fill=initial_fill,
                        demand_scenarios=demand_scenarios,
                        bin_capacity=bin_capacity,
                        t_start=t + 1,
                        H=H,
                        stockout_penalty=stockout_penalty,
                    )
                    if demand_scenarios
                    else 0.0
                )

                inventory_delta = new_inventory_cost - baseline_cost
                total_delta = routing_delta + lambda_inventory * inventory_delta

                if total_delta < best_delta:
                    best_delta = total_delta
                    best_route_idx = r_idx
                    best_pos = pos

        if best_route_idx >= 0:
            # Insert into existing route
            horizon_routes[t][best_route_idx].insert(best_pos, node)
        else:
            # Open a new route (only if demand allows)
            if node_demand <= capacity:
                horizon_routes[t].append([node])

    return horizon_routes
