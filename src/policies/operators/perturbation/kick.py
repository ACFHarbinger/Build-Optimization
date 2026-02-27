"""
Kick Operator Module.

This module implements the 'kick' operator, which destroys a significant portion
of the current solution and then repairs it using a greedy heuristic. This is
typically used to escape deep local optima.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.perturbation.kick import kick
    >>> modified = kick(context, destroy_ratio=0.3)
"""

import random
from typing import Any


def kick(ctx: Any, destroy_ratio: float = 0.2) -> bool:
    """
    Kick operator: destroy a portion of the solution and repair.

    Removes random nodes and reinserts them greedily.

    Args:
        ctx: Context object with routes, node_map, wastes, capacity, etc.
        destroy_ratio: Fraction of nodes to remove (default: 0.2).

    Returns:
        bool: True if the operation was performed, False otherwise.
    """
    all_nodes = list(ctx.node_map.keys())
    n_remove = max(1, int(len(all_nodes) * destroy_ratio))
    if len(all_nodes) < 2:
        return False

    # Select nodes to remove
    removed_nodes = random.sample(all_nodes, min(n_remove, len(all_nodes)))

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
            node_waste = ctx.wastes.get(node, 0)
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
