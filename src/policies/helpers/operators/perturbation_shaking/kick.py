"""
Kick Operator Module.

This module implements the 'kick' operator, which destroys a significant portion
of the current solution and then repairs it using a greedy heuristic. This is
typically used to escape deep local optima.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.kick import kick
    >>> modified = kick(context, destroy_ratio=0.3)
"""

from random import Random
from typing import Any, List, Optional


def kick(ctx: Any, destroy_ratio: float = 0.2, rng: Optional[Random] = None) -> bool:
    """
    Kick operator: destroy a portion of the solution and repair.

    Removes random nodes and reinserts them greedily.

    Args:
        ctx: Context object with routes, node_map, wastes, capacity, etc.
        destroy_ratio: Fraction of nodes to remove (default: 0.2).
        rng: Random number generator.

    Returns:
        bool: True if the operation was performed, False otherwise.
    """
    if rng is None:
        rng = Random()

    all_nodes = list(ctx.node_map.keys())
    n_remove = max(1, int(len(all_nodes) * destroy_ratio))
    if len(all_nodes) < 2:
        return False

    # Select nodes to remove
    removed_nodes = rng.sample(all_nodes, min(n_remove, len(all_nodes)))

    # Remove nodes from routes
    for node in removed_nodes:
        loc = ctx.node_map.get(node)
        if loc:
            ri, pi = loc
            if pi < len(ctx.routes[ri]) and ctx.routes[ri][pi] == node:
                ctx.routes[ri].pop(pi)

    # Remove empty routes
    ctx.routes = [r for r in ctx.routes if r]

    # Rebuild structures after removal
    ctx._build_structures()

    # Greedy reinsertion
    for node in removed_nodes:
        best_cost = float("inf")
        best_ri = -1
        best_pos = -1

        for ri, route in enumerate(ctx.routes):
            current_load = ctx._get_load_cached(ri)
            node_waste = ctx.waste.get(node, 0)
            if current_load + node_waste > ctx.Q:
                continue

            # Try inserting at each position
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost = ctx.d[prev, node] + ctx.d[node, nxt] - ctx.d[prev, nxt]
                if cost < best_cost:
                    best_cost = cost
                    best_ri = ri
                    best_pos = pos

        if best_ri >= 0:
            ctx.routes[best_ri].insert(best_pos, node)
        else:
            # Create new route
            ctx.routes.append([node])

    # Rebuild structures after reinsertion
    ctx._build_structures()
    return True


def kick_profit(
    ctx: Any,
    destroy_ratio: float = 0.2,
    bias: float = 2.0,
    rng: Optional[Random] = None,
) -> bool:
    """
    Profit-driven kick: biased removal of low-profit nodes followed by profit reinsertion.

    Args:
        ctx: Context object (LocalSearch instance).
        destroy_ratio: Fraction of nodes to remove.
        bias: Bias for selecting low-profit nodes (higher = more deterministic).
        rng: Random number generator.

    Returns:
        bool: True if performed.
    """
    if rng is None:
        rng = Random()

    all_nodes = list(ctx.node_map.keys())
    if len(all_nodes) < 2:
        return False

    n_remove = max(1, int(len(all_nodes) * destroy_ratio))

    # Calculate profit contribution for each node
    # Profit(u) = Rev(u) - C * (d(i, u) + d(u, j) - d(i, j))
    node_profits = []
    for node in all_nodes:
        ri, pi = ctx.node_map[node]
        route = ctx.routes[ri]
        prev = route[pi - 1] if pi > 0 else 0
        nxt = route[pi + 1] if pi < len(route) - 1 else 0

        detour = ctx.d[prev, node] + ctx.d[node, nxt] - ctx.d[prev, nxt]
        revenue = ctx.waste.get(node, 0) * ctx.R
        profit = revenue - (detour * ctx.C)
        node_profits.append((node, profit))

    # Sort nodes by profit (ascending)
    node_profits.sort(key=lambda x: x[1])

    # Biased selection of nodes to remove (prefer low profit)
    # We use a simple rank-based selection for robustness
    removed_nodes = []
    candidates = list(node_profits)
    for _ in range(n_remove):
        if not candidates:
            break
        # Pick index with bias towards the start (lower profit)
        idx = int(pow(rng.random(), bias) * len(candidates))
        node, _ = candidates.pop(idx)
        removed_nodes.append(node)

    # Remove nodes
    for node in removed_nodes:
        loc = ctx.node_map.get(node)
        if loc:
            ri, pi = loc
            ctx.routes[ri].pop(pi)
            # Temporarily invalidate node_map entries for this route
            for i in range(pi, len(ctx.routes[ri])):
                n = ctx.routes[ri][i]
                ctx.node_map[n] = (ri, i)
            del ctx.node_map[node]

    # Clean empty routes
    ctx.routes = [r for r in ctx.routes if r]
    ctx._build_structures()

    # Profit-driven reinsertion
    rng.shuffle(removed_nodes)
    _profit_reinsertion(ctx, removed_nodes)

    ctx._build_structures()
    return True


def _profit_reinsertion(ctx: Any, nodes: List[int]) -> None:
    """Helper for profit-driven reinsertion of nodes."""
    for node in nodes:
        best_profit = -float("inf")
        best_ri = -1
        best_pos = -1

        node_waste = ctx.waste.get(node, 0)
        revenue = node_waste * ctx.R

        for ri, route in enumerate(ctx.routes):
            current_load = ctx._get_load_cached(ri)
            if current_load + node_waste > ctx.Q:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost_inc = ctx.d[prev, node] + ctx.d[node, nxt] - ctx.d[prev, nxt]
                profit = revenue - (cost_inc * ctx.C)

                if profit > best_profit:
                    best_profit = profit
                    best_ri = ri
                    best_pos = pos

        if best_ri >= 0 and best_profit > -1e-4:
            ctx.routes[best_ri].insert(best_pos, node)
            ctx._update_map({best_ri})
        else:
            # Check if starting a new route is profitable
            cost_new = ctx.d[0, node] + ctx.d[node, 0]
            profit_new = revenue - (cost_new * ctx.C)
            if profit_new > -1e-4:
                ctx.routes.append([node])
                ctx._build_structures()
            # Else: node remains unvisited

    ctx._build_structures()
    return
