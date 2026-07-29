"""
Inter-Day Shift Operator.
"""

import random

import numpy as np

from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual import (
    Individual,
)
from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.split import (
    compute_daily_loads,
    split_day,
)


def inter_day_shift(
    ind: Individual,
    base_wastes: np.ndarray,
    daily_increments: np.ndarray,
    dist: np.ndarray,
    capacity: float,
    n_vehicles: int,
    T: int,
) -> bool:
    """
    Inter-Day Shift Operator.

    Tries to shift a node i from an active day t to an inactive day t'.
    Flips the bits in p_i to maintain open unconstrained search.
    Re-evaluates incremental loads and splits.
    If fitness improves, commits the change.

    Returns True if an improvement was found.
    """
    N = len(ind.patterns)

    nodes = list(range(1, N))
    random.shuffle(nodes)

    original_fit = ind.fit

    for node in nodes:
        active_days = [t for t in range(T) if ind.is_active(node, t)]
        inactive_days = [t for t in range(T) if not ind.is_active(node, t)]

        if not active_days or not inactive_days:
            continue

        # Try shifting one visit
        for from_day in active_days:
            for to_day in inactive_days:
                # 1. Morph Pattern
                orig_pattern = ind.patterns[node]
                ind.set_active(node, from_day, False)
                ind.set_active(node, to_day, True)

                # 2. Morph Giant Tours
                orig_gt_from = ind.giant_tours[from_day].copy()
                orig_gt_to = ind.giant_tours[to_day].copy()

                # Remove from from_day
                ind.giant_tours[from_day] = np.array([n for n in orig_gt_from if n != node], dtype=int)

                # Insert randomly into to_day
                insert_idx = random.randint(0, len(orig_gt_to))
                new_to = list(orig_gt_to)
                new_to.insert(insert_idx, node)
                ind.giant_tours[to_day] = np.array(new_to, dtype=int)

                # 3. Re-evaluate Loads (Only needs to care about horizon,
                # but we recompute all purely to ensure state tracks incrementals correctly)
                loads = compute_daily_loads(ind.patterns, base_wastes, daily_increments, T)

                new_cost = 0.0
                new_violations = 0.0
                new_routes = []

                for t in range(T):
                    r, c, v = split_day(ind.giant_tours[t], loads[t], dist, capacity, n_vehicles)
                    new_cost += c
                    new_violations += v
                    new_routes.append(r)

                w_q = 100.0  # static penalty for delta-Q
                new_fit = new_cost + w_q * new_violations

                if new_fit < original_fit - 1e-4:
                    # Keep improvement
                    ind.cost = new_cost
                    ind.capacity_violations = new_violations
                    ind.fit = new_fit
                    ind.routes = new_routes
                    ind.is_feasible = new_violations == 0.0
                    return True

                else:
                    # Revert
                    ind.patterns[node] = orig_pattern
                    ind.giant_tours[from_day] = orig_gt_from
                    ind.giant_tours[to_day] = orig_gt_to

    return False
