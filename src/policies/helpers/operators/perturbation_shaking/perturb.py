"""
Perturb Operator Module.

This module implements the 'perturb' operator, which performs a series of random
swaps to shake up the current solution and escape local optima.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.perturb import perturb
    >>> changed = perturb(context, k=5)
"""

from random import Random
from typing import Any, Optional


def perturb(
    ctx: Any,
    k: int = 3,
    prob_unvisited: float = 0.0,
    rng: Optional[Random] = None,
) -> bool:
    """
    Perturbation operator: perform k random swaps to escape local optima.

    Attempts to perform `k` random swaps. If prob_unvisited > 0, some swaps
    may involve currently unvisited nodes.

    Args:
        ctx: Context object with routes, node_map, wastes, capacity, etc.
        k: Number of swaps to perform (default: 3).
        prob_unvisited: Probability of swapping with an unvisited node.
        rng: Random number generator.

    Returns:
        bool: True if any swap was performed, False otherwise.
    """
    if rng is None:
        rng = Random()

    n_nodes = len(ctx.d) - 1
    visited = set(ctx.node_map.keys())
    unvisited = sorted(list(set(range(1, n_nodes + 1)) - visited))

    performed = False
    for _ in range(k):
        # Decide whether to swap between visited or visited-unvisited
        if unvisited and rng.random() < prob_unvisited:
            # Swap a visited node with an unvisited one
            u = rng.choice(list(visited))
            v = rng.choice(unvisited)

            u_loc = ctx.node_map.get(u)
            if not u_loc:
                continue
            r_u, p_u = u_loc

            dem_u = ctx.waste.get(u, 0)
            dem_v = ctx.waste.get(v, 0)

            # Check capacity
            current_load = sum(ctx.waste.get(node, 0) for node in ctx.routes[r_u])
            if current_load - dem_u + dem_v <= ctx.Q:
                ctx.routes[r_u][p_u] = v
                ctx._update_map({r_u})
                visited.remove(u)
                visited.add(v)
                unvisited.remove(v)
                unvisited.append(u)
                performed = True
        else:
            # Standard random swap between visited nodes
            all_visited = list(visited)
            if len(all_visited) < 2:
                continue
            u, v = rng.sample(all_visited, 2)
            u_loc = ctx.node_map.get(u)
            v_loc = ctx.node_map.get(v)
            if not u_loc or not v_loc:
                continue

            r_u, p_u = u_loc
            r_v, p_v = v_loc

            if r_u == r_v:
                if abs(p_u - p_v) > 1:
                    ctx.routes[r_u][p_u], ctx.routes[r_v][p_v] = (
                        ctx.routes[r_v][p_v],
                        ctx.routes[r_u][p_u],
                    )
                    ctx._update_map({r_u})
                    performed = True
            else:
                dem_u = ctx.waste.get(u, 0)
                dem_v = ctx.waste.get(v, 0)
                if (
                    ctx._get_load_cached(r_u) - dem_u + dem_v <= ctx.Q
                    and ctx._get_load_cached(r_v) - dem_v + dem_u <= ctx.Q
                ):
                    ctx.routes[r_u][p_u] = v
                    ctx.routes[r_v][p_v] = u
                    ctx._update_map({r_u, r_v})
                    performed = True

    return performed


def perturb_profit(
    ctx: Any,
    k: int = 3,
    prob_unvisited: float = 0.5,
    rng: Optional[Random] = None,
) -> bool:
    """
    Profit-driven perturbation: attempt k swaps with a focus on unvisited nodes.

    Args:
        ctx: Context object (LocalSearch instance).
        k: Number of perturbation attempts.
        prob_unvisited: Probability of attempting a swap with an unvisited node.
        rng: Random number generator.

    Returns:
        bool: True if any change was applied.
    """
    if rng is None:
        rng = Random()

    n_nodes = len(ctx.d) - 1
    visited = set(ctx.node_map.keys())
    unvisited = sorted(list(set(range(1, n_nodes + 1)) - visited))

    performed = False
    for _ in range(k):
        # Decide whether to swap between visited or visited-unvisited
        if unvisited and rng.random() < prob_unvisited:
            # Swap a visited node with an unvisited one
            u = rng.choice(list(visited))
            v = rng.choice(unvisited)

            u_loc = ctx.node_map.get(u)
            if not u_loc:
                continue
            r_u, p_u = u_loc

            dem_u = ctx.waste.get(u, 0)
            dem_v = ctx.waste.get(v, 0)

            # Check capacity
            current_load = sum(ctx.waste.get(node, 0) for node in ctx.routes[r_u])
            if current_load - dem_u + dem_v <= ctx.Q:
                # Calculate profit delta
                # delta_dist = d(prev, v) + d(v, nxt) - (d(prev, u) + d(u, nxt))
                route = ctx.routes[r_u]
                prev = route[p_u - 1] if p_u > 0 else 0
                nxt = route[p_u + 1] if p_u < len(route) - 1 else 0

                delta_dist = ctx.d[prev, v] + ctx.d[v, nxt] - (ctx.d[prev, u] + ctx.d[u, nxt])
                delta_rev = (dem_v - dem_u) * ctx.R
                profit_gain = delta_rev - (delta_dist * ctx.C)

                # For perturbation, we can accept slightly negative profit for diversification
                # but we prefer non-negative or strictly positive for "profit-driven"
                if profit_gain > -1e-4:
                    ctx.routes[r_u][p_u] = v
                    ctx._update_map({r_u})
                    visited.remove(u)
                    visited.add(v)
                    unvisited.remove(v)
                    unvisited.append(u)
                    performed = True
        else:
            # Standard random swap between visited nodes (from perturb)
            all_visited = list(visited)
            if len(all_visited) < 2:
                continue
            u, v = rng.sample(all_visited, 2)
            u_loc = ctx.node_map.get(u)
            v_loc = ctx.node_map.get(v)
            if not u_loc or not v_loc:
                continue

            r_u, p_u = u_loc
            r_v, p_v = v_loc

            if r_u == r_v:
                if abs(p_u - p_v) > 1:
                    ctx.routes[r_u][p_u], ctx.routes[r_v][p_v] = (
                        ctx.routes[r_v][p_v],
                        ctx.routes[r_u][p_u],
                    )
                    ctx._update_map({r_u})
                    performed = True
            else:
                dem_u = ctx.waste.get(u, 0)
                dem_v = ctx.waste.get(v, 0)
                if (
                    ctx._get_load_cached(r_u) - dem_u + dem_v <= ctx.Q
                    and ctx._get_load_cached(r_v) - dem_v + dem_u <= ctx.Q
                ):
                    ctx.routes[r_u][p_u] = v
                    ctx.routes[r_v][p_v] = u
                    ctx._update_map({r_u, r_v})
                    performed = True

    return performed
