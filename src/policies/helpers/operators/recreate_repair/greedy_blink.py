"""
Greedy Blink Insertion Operator Module.

This module implements the 'Greedy with Blinks' insertion heuristic based on
the Slack Induction by String Removal (SISR) metaheuristic.

It contains:
    1. `greedy_insertion_with_blinks`: A standard distance-minimizing operator.
    2. `greedy_profit_insertion_with_blinks`: A VRPP-specific profit-maximizing
       operator that utilizes speculative seeding and post-insertion pruning.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.greedy_blink import greedy_insertion_with_blinks
    >>> routes = greedy_insertion_with_blinks(routes, removed, dist_matrix, wastes, capacity, blink_rate=0.1)
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def greedy_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    blink_rate: float = 0.1,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Standard Greedy insertion with randomized skips ('blinks').
    Optimizes strictly for minimum distance/cost.

    Implements the *Greedy with Blinks* repair operator from Christiaens &
    Vanden Berghe (2020), *Slack Induction by String Removals for Vehicle
    Routing Problems* (SISR), §3.2 (Insertion with blinks):

    For each unassigned node, the sorted list of candidate positions is scanned
    from best to worst.  Each position is **skipped** ("blinked") independently
    with probability ``blink_rate``.  The first non-blinked position is selected.
    If **all** positions are blinked, the algorithm reverts to the **globally
    best** (lowest-cost) position to guarantee that every node is eventually
    inserted.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to reinsert.
        dist_matrix: Distance matrix.
        wastes: Node demands.
        capacity: Vehicle capacity.
        blink_rate: Probability of skipping a candidate position (\u03b2 in the
            paper).
        mandatory_nodes: List of nodes that must be inserted.
        rng: Random number generator.
        expand_pool: If True, reconstructs the unassigned pool from all unvisited nodes.

    Returns:
        Routes with nodes reinserted based on minimum cost.
    """
    mandatory_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if rng is None:
        rng = Random()

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = list(removed_nodes)

    # Shuffle for randomness, then stable sort mandatory nodes to the front
    rng.shuffle(unassigned)
    unassigned.sort(key=lambda x: 0 if x in mandatory_set else 1)
    for node in unassigned:
        node_waste = wastes.get(node, 0)
        is_man = node in mandatory_set
        options = []

        # Check existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                options.append((cost_delta, r_idx, pos))

        # Check new route
        new_route_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        options.append((new_route_cost, len(routes), 0))
        if not options:
            if is_man:
                routes.append([node])
                loads.append(node_waste)
            continue

        # Sort options by Cost (Ascending - Lowest cost is best)
        options.sort(key=lambda x: x[0])

        # True Option Blink Logic (Christiaens & Vanden Berghe 2020, §3.2)
        # Scan sorted options; accept the first non-blinked position.
        # If ALL positions are blinked, fall back to the best option (options[0])
        # so that the node is never left unassigned.
        best_selection = options[0]  # guaranteed fallback = best option
        for opt in options:
            if rng.random() >= blink_rate:
                best_selection = opt
                break

        # Apply the chosen move
        cost, r_idx, pos = best_selection
        if r_idx == len(routes):
            routes.append([node])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, node)
            loads[r_idx] += node_waste

    return routes


def greedy_profit_insertion_with_blinks(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    blink_rate: float = 0.1,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Greedy profit-driven insertion with randomized skips ('blinks').
    Includes Speculative Seeding and Economic Pruning for VRPP.

    Implements the *Greedy with Blinks* repair operator from Christiaens &
    Vanden Berghe (2020) adapted for profit-maximisation (VRPP):
    - Candidates are sorted descending by marginal profit.
    - Each candidate is skipped with probability ``blink_rate``.
    - If **all** candidates are blinked, the algorithm falls back to the
      **best** (highest-profit) option, not the worst, to ensure the node
      is always placed.
    - A new route is opened (Speculative Seeding) when the node's standalone
      profit passes the seed hurdle: ``profit ≥ -0.5 * round_trip_cost * C``.
    - Economic Pruning (`prune_unprofitable_routes`) removes any seeded routes
      that remain unprofitable at the end.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier (per waste unit).
        C: Cost multiplier (per distance unit).
        blink_rate: Probability \u03b2 of skipping a candidate position.
        mandatory_nodes: List of mandatory node indices.
        rng: Random number generator.
        expand_pool: If True, reconstructs the unassigned pool from all unvisited nodes.

    Returns:
        List[List[int]]: Updated routes, pruned of unprofitable excursions.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

    if rng is None:
        rng = Random()

    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    rng.shuffle(unassigned)

    for node in unassigned:
        node_waste = wastes.get(node, 0)
        revenue = node_waste * R
        is_mandatory = node in mandatory_nodes_set

        candidates = []  # List of (profit, route_idx, position)

        # Evaluate existing routes
        for r_idx, route in enumerate(routes):
            if loads[r_idx] + node_waste > capacity:
                continue

            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost_delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                profit = revenue - (cost_delta * C)

                # Check profitability hurdle
                if is_mandatory or profit > -1e-4:
                    candidates.append((profit, r_idx, pos))

        # Evaluate new route (with speculative seed)
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - (new_cost * C)

        # Allow up to 50% of the return-trip cost to be covered by synergy later.
        seed_hurdle = -0.5 * (new_cost * C)
        if is_mandatory or new_profit >= seed_hurdle:
            candidates.append((new_profit, len(routes), 0))

        # Emergency fallback if no valid options exist
        if not candidates:
            if is_mandatory:
                routes.append([node])
                loads.append(node_waste)
            continue

        # Sort candidates by Profit (Descending - Highest profit is best)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # True Option Blink Logic (Christiaens & Vanden Berghe 2020, §3.2)
        # Scan sorted candidates (best first); accept first non-blinked position.
        # If ALL are blinked, fall back to candidates[0] (the best option).
        selected = candidates[0]  # guaranteed fallback = best option
        for cand in candidates:
            if rng.random() >= blink_rate:
                selected = cand
                break

        # Apply insertion
        profit, r_idx, pos = selected
        if r_idx == len(routes):
            routes.append([node])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, node)
            loads[r_idx] += node_waste

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)
