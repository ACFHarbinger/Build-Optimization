"""
Historical Node Removal Operator Module.

Uses historical memory to penalize nodes that are frequently found in
high-cost solutions.  Removes the ``k`` nodes with the worst historical
scores from their routes.

Also includes profit-based historical removal for VRPP problems.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.historical import historical_removal
    >>> history = {1: 12.5, 2: 8.0, 3: 15.3, ...}
    >>> routes, removed = historical_removal(routes, n_remove=5, history=history)
    >>> from logic.src.policies.helpers.operators.destroy.historical import historical_profit_removal
    >>> routes, removed = historical_profit_removal(routes, n_remove=5, history=history,
    ...                                             dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np


def historical_removal(
    routes: List[List[int]],
    n_remove: int,
    history: Dict[int, float],
    rng: Optional[Random] = None,
    noise: float = 0.1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with the worst historical cost scores.

    **Paper Reference**: Pisinger & Ropke (2007), §5.1.6 — *Historical
    Node-Pair Removal* (named *neighbor graph removal* in the earlier 2006
    paper).

    **Paper mechanics**: A weight ``f*(u,v)`` is associated with each directed
    arc ``(u, v)``.  It records the best objective value seen so far in any
    solution that used arc ``(u, v)`` — initially set to ``+inf``.  Whenever
    a new solution is found, all arc weights are updated with
    ``f*(u,v) = min(f*(u,v), f(solution))``.  A node's score is the sum of
    weights of its incident arcs in the current solution.  **High score ⟹
    incident arcs tend to appear in costly solutions ⟹ node is a good
    removal candidate.**

    **Expected ``history`` dict semantics**: map each node ``i`` to a
    non-negative penalty representing how costly its incident arc has been
    historically.  Higher values indicate that the node is more likely to be
    misplaced.  The caller is responsible for maintaining and updating these
    scores after each ALNS iteration using the paper's update rule above.

    Scores each node by its historical penalty (higher = worse).  A small
    random noise term breaks ties and introduces diversity.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        history: Dictionary mapping node IDs to their historical cost scores
                 (higher = worse placement; maintained by the caller across
                 ALNS iterations per §5.1.6 of Pisinger & Ropke 2007).
        rng: Random number generator.
        noise: Noise amplitude (fraction of max score) added for diversity.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random()

    # Collect all nodes with their scores
    scored: List[Tuple[float, int, int, int]] = []  # (score, node, r_idx, pos)
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            base_score = history.get(node, 0.0)
            scored.append((base_score, node, r_idx, pos))

    if not scored:
        return routes, []

    # Add noise
    max_score = max(s for s, _, _, _ in scored) if scored else 1.0
    noise_amp = noise * max(max_score, 1e-6)
    scored_noisy = [
        (score + rng.uniform(-noise_amp, noise_amp), node, r_idx, pos) for score, node, r_idx, pos in scored
    ]

    # Sort descending by score (worst nodes first)
    scored_noisy.sort(reverse=True)

    n_remove = min(n_remove, len(scored_noisy))
    to_remove = scored_noisy[:n_remove]

    # Sort by (route_idx, position) descending for safe removal
    to_remove.sort(key=lambda x: (x[2], x[3]), reverse=True)

    removed: List[int] = []
    for _, node, r_idx, pos in to_remove:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed


def historical_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    history: Dict[int, float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    alpha: float = 0.5,
    rng: Optional[Random] = None,
    noise: float = 0.1,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove nodes with worst combined historical score and current profit (VRPP).

    Combines historical penalty with current profit evaluation to remove
    nodes that have been problematic in the past AND have low current profit.

    Args:
        routes: Current solution (list of routes).
        n_remove: Number of nodes to remove.
        history: Dictionary mapping node IDs to their historical cost scores
                 (moving averages across iterations).
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        alpha: Weight for historical score (1-alpha for profit score).
               alpha=1.0 uses only history, alpha=0.0 uses only profit.
        rng: Random number generator.
        noise: Noise amplitude (fraction of max score) added for diversity.

    Returns:
        Tuple of (modified routes, list of removed node IDs).
    """
    if rng is None:
        rng = Random()

    # 1. Pre-calculate node profits with marginal formula
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

    # 2. Normalize and Combine
    profit_vals = list(node_profits.values())
    min_p, max_p = min(profit_vals), max(profit_vals)
    p_range = max_p - min_p if max_p != min_p else 1.0

    hist_vals = [history.get(n, 0.0) for n in all_nodes]
    min_h, max_h = min(hist_vals) if hist_vals else 0.0, max(hist_vals) if hist_vals else 1.0
    h_range = max_h - min_h if max_h != min_h else 1.0

    scored: List[Tuple[float, int, int, int]] = []
    for r_idx, pos, node in [(node_map[n][0], node_map[n][1], n) for n in all_nodes]:
        # Normalize historical score (higher is worse)
        h_norm = (history.get(node, 0.0) - min_h) / h_range
        # Normalize profit (inverted: lower profit = higher score)
        p_norm = (max_p - node_profits[node]) / p_range

        combined = alpha * h_norm + (1 - alpha) * p_norm
        scored.append((combined, node, r_idx, pos))

    # 3. Randomize and Select
    max_score = max(s for s, _, _, _ in scored) if scored else 1.0
    noise_amp = noise * max(max_score, 1e-6)
    scored_noisy = [(s + rng.uniform(-noise_amp, noise_amp), n, r, p) for s, n, r, p in scored]
    scored_noisy.sort(reverse=True)

    n_remove = min(n_remove, len(scored_noisy))
    targets = scored_noisy[:n_remove]
    targets.sort(key=lambda x: (x[2], x[3]), reverse=True)

    removed: List[int] = []
    for _, node, r_idx, pos in targets:
        if pos < len(routes[r_idx]) and routes[r_idx][pos] == node:
            routes[r_idx].pop(pos)
            removed.append(node)

    routes = [r for r in routes if r]
    return routes, removed
