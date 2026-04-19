"""
Multi-Period Repair Operators for MPVRPP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.recreate_repair.forward_looking import (
    _compute_routing_delta,
    _simulate_inventory_forward,
)


def greedy_horizon_insertion(
    horizon_routes: List[List[List[int]]],
    removed: List[Tuple[int, int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    scenario_tree: Optional[Any] = None,
    use_stochastic: bool = False,
    stockout_penalty: float = 500.0,
    look_ahead_days: int = 3,
) -> List[List[List[int]]]:
    """Insert removed nodes into the best (day, route, pos) across the horizon.

    Args:
        horizon_routes: Full T-day plan.
        removed: Removed (node, day) pairs. Note: target day is ignored
            if we search the full horizon.
        dist_matrix: Distance matrix.
        wastes: Current fill levels.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        scenario_tree: Optional ScenarioTree for lookahead.
        use_stochastic: If True, uses forward-looking inventory simulation.
        stockout_penalty: Penalty for overflow.
        look_ahead_days: Lookahead H.
    """
    T = len(horizon_routes)
    # Extract nodes (ignore the days assigned by destroy ops for true horizon repair)
    nodes_to_insert = [node for node, _ in removed]

    for node in nodes_to_insert:
        best_gain = -float("inf")
        best_day = -1
        best_route_idx = -1
        best_pos = -1

        # Evaluate across all days
        for t in range(T):
            # Baseline inventory if we skip day t (simplified)
            # For a true multi-day repair, inventory depends on other days' visits.
            # We use forward_looking logic if use_stochastic is True.

            for r_idx, route in enumerate(horizon_routes[t]):
                # Capacity check
                load = sum(wastes.get(n, 0.0) for n in route)
                if load + wastes.get(node, 0.0) > capacity:
                    continue

                for pos in range(len(route) + 1):
                    routing_delta = _compute_routing_delta(route, node, pos, dist_matrix, C)
                    revenue = wastes.get(node, 0.0) * R

                    if use_stochastic and scenario_tree:
                        # Simplified FL check: profit = revenue - routing_cost - E[Overflow]
                        # Actually we use the logic from forward_looking_insertion if toggled
                        inv_cost = _simulate_inventory_forward(
                            node=node,
                            visit_days=[t],  # Simplified: only consider this visit
                            initial_fill=wastes.get(node, 0.0),
                            demand_scenarios=_get_scenarios(scenario_tree, node, t, look_ahead_days),
                            bin_capacity=100.0,
                            t_start=t + 1,
                            H=look_ahead_days,
                            stockout_penalty=stockout_penalty,
                        )
                        gain = revenue - routing_delta - inv_cost
                    else:
                        gain = revenue - routing_delta

                    if gain > best_gain:
                        best_gain = gain
                        best_day = t
                        best_route_idx = r_idx
                        best_pos = pos

        if best_day >= 0:
            horizon_routes[best_day][best_route_idx].insert(best_pos, node)
        else:
            # Optionally seed new route on day with highest demand (or first day)
            horizon_routes[0].append([node])

    return horizon_routes


def regret_k_temporal_insertion(
    horizon_routes: List[List[List[int]]],
    removed: List[Tuple[int, int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    k: int = 2,
    scenario_tree: Optional[Any] = None,
    use_stochastic: bool = False,
    stockout_penalty: float = 500.0,
) -> List[List[List[int]]]:
    """Regret-k insertion where regret is calculated across DIFFERENT DAYS.

    Prioritizes nodes whose profit difference between the best day and k-th
    best day is highest.
    """
    T = len(horizon_routes)
    nodes_to_insert = [node for node, _ in removed]

    while nodes_to_insert:
        node_regrets = []
        for node in nodes_to_insert:
            day_profits = []
            for t in range(T):
                # Find best insertion on day t
                best_t_profit = -float("inf")
                best_t_pos = (-1, -1)
                for r_idx, route in enumerate(horizon_routes[t]):
                    load = sum(wastes.get(n, 0.0) for n in route)
                    if load + wastes.get(node, 0.0) > capacity:
                        continue
                    for pos in range(len(route) + 1):
                        routing_delta = _compute_routing_delta(route, node, pos, dist_matrix, C)
                        revenue = wastes.get(node, 0.0) * R
                        profit = revenue - routing_delta
                        if profit > best_t_profit:
                            best_t_profit = profit
                            best_t_pos = (r_idx, pos)
                day_profits.append((best_t_profit, t, best_t_pos))

            day_profits.sort(key=lambda x: x[0], reverse=True)

            # Temporal Regret: Profit(Best Day) - Profit(k-th Best Day)
            regret = day_profits[0][0] - day_profits[k - 1][0] if len(day_profits) >= k else day_profits[0][0] + 1000.0
            node_regrets.append((regret, node, day_profits[0]))

        # Pick node with max regret
        node_regrets.sort(key=lambda x: x[0], reverse=True)
        _, chosen_node, (profit, t, (r_idx, pos)) = node_regrets[0]

        if t >= 0 and r_idx >= 0:
            horizon_routes[t][r_idx].insert(pos, chosen_node)

        nodes_to_insert.remove(chosen_node)

    return horizon_routes


def stochastic_aware_insertion(
    horizon_routes: List[List[List[int]]],
    removed: List[Tuple[int, int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    scenario_tree: Optional[Any] = None,
    stockout_penalty: float = 500.0,
) -> List[List[List[int]]]:
    """Insertion guided by expected future fill levels E[w_i,t].

    Bias insertion toward days where the node is expected to be near capacity.
    """
    if scenario_tree is None:
        return greedy_horizon_insertion(horizon_routes, removed, dist_matrix, wastes, capacity, R, C)

    T = len(horizon_routes)
    nodes_to_insert = [node for node, _ in removed]

    for node in nodes_to_insert:
        # 1. Compute E[w_i,t] for all t
        expected_fills = []
        for t_day in range(T):
            # Try to get data from scenario tree if available
            if hasattr(scenario_tree, "nodes"):
                # tree.py structure: Dict[id, ScenarioNode]
                day_nodes = [n for n in scenario_tree.nodes.values() if n.day == t_day]
                if day_nodes:
                    avg_fill = sum(n.realization.get(node, 0.0) for n in day_nodes) / len(day_nodes)
                else:
                    avg_fill = wastes.get(node, 0.0)
            elif hasattr(scenario_tree, "get_scenarios_at_day"):
                # prediction.py structure
                day_scenarios = scenario_tree.get_scenarios_at_day(t_day)
                if day_scenarios:
                    avg_fill = sum(float(getattr(s, "wastes", [0.0] * node)[node - 1]) for s in day_scenarios) / len(
                        day_scenarios
                    )
                else:
                    avg_fill = wastes.get(node, 0.0)
            else:
                avg_fill = wastes.get(node, 0.0)
            expected_fills.append(avg_fill)

        # 2. Greedy insertion but scale profit by expected occupancy (urgency)
        best_score = -float("inf")
        best_day = -1
        best_route = -1
        best_pos = -1

        for t in range(T):
            urgency_weight = expected_fills[t] / 100.0  # Normalized 0-1
            for r_idx, route in enumerate(horizon_routes[t]):
                load = sum(wastes.get(n, 0.0) for n in route)
                if load + wastes.get(node, 0.0) > capacity:
                    continue
                for pos in range(len(route) + 1):
                    routing_delta = _compute_routing_delta(route, node, pos, dist_matrix, C)
                    profit = (wastes.get(node, 0.0) * R) - routing_delta
                    # Weighted score: profit scaled by urgency
                    score = profit * (1.0 + urgency_weight)
                    if score > best_score:
                        best_score = score
                        best_day = t
                        best_route = r_idx
                        best_pos = pos

        if best_day >= 0:
            horizon_routes[best_day][best_route].insert(best_pos, node)

    return horizon_routes


def _get_scenarios(tree: Any, node: int, t: int, H: int) -> List[List[float]]:
    """Helper to extract scenario sequences from tree starting from day t.

    A scenario is a path of length H (from t+1 to t+H).
    """
    if tree is None:
        return []

    # Case 1: tree.py structure (Dict[id, ScenarioNode])
    if hasattr(tree, "nodes"):
        return _get_scenarios_from_ef(tree, node, t, H)

    # Case 2: prediction.py structure (root-based ScenarioTreeNode)
    if hasattr(tree, "root"):
        return _get_scenarios_from_prediction(tree, node, t, H)

    return []


def _get_scenarios_from_ef(tree: Any, node: int, t: int, H: int) -> List[List[float]]:
    """Helper for ID-based tree traversal (tree.py)."""
    start_nodes = [n for n in tree.nodes.values() if n.day == t]
    if not start_nodes:
        return []

    all_sequences: List[List[float]] = []

    def dfs_ef(curr_node: Any, current_path: List[float]):
        if len(current_path) == H:
            all_sequences.append(current_path)
            return

        if not curr_node.children_ids:
            all_sequences.append(current_path + [0.0] * (H - len(current_path)))
            return

        for child_id in curr_node.children_ids:
            if child_id in tree.nodes:
                child = tree.nodes[child_id]
                val = child.realization.get(node, 0.0)
                dfs_ef(child, current_path + [val])

    for start in start_nodes:
        dfs_ef(start, [])
    return all_sequences


def _get_scenarios_from_prediction(tree: Any, node: int, t: int, H: int) -> List[List[float]]:
    """Helper for root-based tree traversal (prediction.py)."""
    from logic.src.pipeline.simulations.bins.prediction import ScenarioTreeNode

    start_nodes_p: List[ScenarioTreeNode] = []

    def find_nodes(curr: ScenarioTreeNode, target_d: int):
        if curr.day == target_d:
            start_nodes_p.append(curr)
            return
        for child in curr.children:
            find_nodes(child, target_d)

    find_nodes(tree.root, t)

    all_sequences_p: List[List[float]] = []

    def dfs_p(curr: ScenarioTreeNode, current_path: List[float]):
        if len(current_path) == H:
            all_sequences_p.append(current_path)
            return

        if not curr.children:
            all_sequences_p.append(current_path + [0.0] * (H - len(current_path)))
            return

        for child in curr.children:
            val = float(child.wastes[node - 1]) if node - 1 < len(child.wastes) else 0.0
            dfs_p(child, current_path + [val])

    for start in start_nodes_p:
        dfs_p(start, [])
    return all_sequences_p
