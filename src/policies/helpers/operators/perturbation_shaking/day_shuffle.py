"""
Day-shuffle perturbation: randomly reassign a subset of bins to other days.
"""

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext


def day_shuffle(
    plan: List[List[int]],
    problem: ProblemContext,
    rng: np.random.Generator,
    k: int = 3,
    max_attempts: int = 20,
) -> Tuple[List[List[int]], int]:
    """
    Perturb the multi-period plan by randomly reassigning nodes to different days.

    This operator selects exactly `k` bins (node IDs) across the entire D-day
    planning horizon and attempts to move each of them to a randomly chosen
    target day. It is a powerful "cross-day" perturbation that can break
    period-partitioning local optima by forcing customers to be served on
    completely different days.

    For each selected node, the operator:
    1. Removes it from its current day.
    2. Samples a random permutation of all available days in the horizon.
    3. Iterates through the days and attempts to insert the node at the best
       (lowest distance increase) position in each day's route.
    4. Validates capacity feasibility on the target day using day-0 fill levels.
    5. If no feasible target day is found within the specified `max_attempts`
       (or if all days are exhausted), the node is returned to its original day.

    Args:
        plan:         Initial D-day routing plan (List of List of nodes).
        problem:      The day-0 ProblemContext containing the master capacity
                      and fill-level estimates.
        rng:          NumPy Generator instance for deterministic randomness.
        k:            The number of nodes to attempt to shuffle.
        max_attempts: The maximum number of target days to try per node before
                      giving up and reverting the move.

    Returns:
        Tuple[List[List[int]], int]:
            - The perturbed D-day plan.
            - The index of the earliest day modified (to allow efficient
              cascaded re-simulation of fill levels).

    Complexity:
        O(k * D * max_route_length) for finding the best insertion point.
    """
    Q = problem.capacity
    w = problem.wastes
    D = len(plan)

    # Collect all (bin, day) pairs
    all_pairs = [(bin_id, d) for d, route in enumerate(plan) for bin_id in route]
    if not all_pairs:
        return plan, 0

    k_actual = min(k, len(all_pairs))
    chosen_indices = rng.choice(len(all_pairs), size=k_actual, replace=False)
    chosen = [all_pairs[i] for i in chosen_indices]

    new_plan = [list(r) for r in plan]
    loads = [sum(w.get(n, 0.0) for n in r) for r in new_plan]
    earliest = D

    for bin_id, src_day in chosen:
        if bin_id not in new_plan[src_day]:
            continue

        bin_waste = w.get(bin_id, 0.0)
        new_plan[src_day] = [n for n in new_plan[src_day] if n != bin_id]
        loads[src_day] -= bin_waste

        # Try random target days
        day_order = list(rng.permutation(D))
        placed = False
        for tgt_day in day_order:
            if tgt_day == src_day:
                continue
            if loads[tgt_day] + bin_waste <= Q:
                # Insert at best position
                route = new_plan[tgt_day]
                dm = problem.distance_matrix
                best_cost = float("inf")
                best_pos = 0
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    cost = dm[prev, bin_id] + dm[bin_id, nxt] - dm[prev, nxt]
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                new_plan[tgt_day].insert(best_pos, bin_id)
                loads[tgt_day] += bin_waste
                earliest = min(earliest, min(src_day, tgt_day))
                placed = True
                break

        if not placed:
            # Return bin to original day if no target found
            new_plan[src_day].append(bin_id)
            loads[src_day] += bin_waste

    return new_plan, min(earliest, D - 1)
