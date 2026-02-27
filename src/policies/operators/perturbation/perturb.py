"""
Perturb Operator Module.

This module implements the 'perturb' operator, which performs a series of random
swaps to shake up the current solution and escape local optima.

Attributes:
    None

Example:
    >>> from logic.src.policies.operators.perturbation.perturb import perturb
    >>> changed = perturb(context, k=5)
"""

import random
from typing import Any


def perturb(ctx: Any, k: int = 3) -> bool:
    """
    Perturbation operator: perform k random swaps to escape local optima.

    Attempts to perform `k` random swaps between nodes. Swaps are forced
    even if they don't improve the objective, provided they are feasible.

    Args:
        ctx: Context object with routes, node_map, wastes, capacity, etc.
        k: Number of swaps to perform (default: 3).

    Returns:
        bool: True if any swap was performed, False otherwise.
    """
    all_nodes = list(ctx.node_map.keys())
    if len(all_nodes) < 2:
        return False

    performed = False
    for _ in range(k):
        if len(all_nodes) < 2:
            break
        u, v = random.sample(all_nodes, 2)
        u_loc = ctx.node_map.get(u)
        v_loc = ctx.node_map.get(v)
        if not u_loc or not v_loc:
            continue

        r_u, p_u = u_loc
        r_v, p_v = v_loc

        # Force swap regardless of improvement
        if r_u == r_v:
            # Same route swap
            if abs(p_u - p_v) > 1:
                ctx.routes[r_u][p_u], ctx.routes[r_v][p_v] = (
                    ctx.routes[r_v][p_v],
                    ctx.routes[r_u][p_u],
                )
                ctx._update_map({r_u})
                performed = True
        else:
            # Inter-route swap
            dem_u = ctx.wastes.get(u, 0)
            dem_v = ctx.wastes.get(v, 0)
            if (
                ctx._get_load_cached(r_u) - dem_u + dem_v <= ctx.Q
                and ctx._get_load_cached(r_v) - dem_v + dem_u <= ctx.Q
            ):
                ctx.routes[r_u][p_u] = v
                ctx.routes[r_v][p_v] = u
                ctx._update_map({r_u, r_v})
                performed = True

    return performed
