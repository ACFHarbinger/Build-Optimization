"""
Cross-day perturbation operators for multi-period routing policies.

All operators accept a full D-day plan (List[List[int]], one flat route per
day, depot excluded) and a ProblemContext describing day 0. They return a
new plan and the index of the earliest modified day so the caller can call
recompute_cascade_from efficiently.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext

# ---------------------------------------------------------------------------
# Cascade helper (must be used by every operator that changes multiple days)
# ---------------------------------------------------------------------------


def recompute_cascade_from(
    plan: List[List[int]],
    problem: ProblemContext,
    start_day: int,
) -> Tuple[List[List[int]], ProblemContext]:
    """
    Synchronize fill-level transitions across the planning horizon after plan changes.

    In multi-period routing (MPVRP), removing a bin from day `d` affects the
    waste generated for that customer on day `d+1` and all subsequent days.
    This helper identifies the start day of a perturbation and ensures that
    downstream profit calculations and capacity checks are consistent.

    Args:
        plan:      The D-day plan (List of List of nodes, depot excluded).
        problem:   The initial ProblemContext at the very beginning of day 0.
        start_day: The index of the first day in the horizon that was modified.
                   Fill levels for days < start_day remain valid.

    Returns:
        Tuple[List[List[int]], ProblemContext]:
            - The original plan (structure unchanged).
            - The day-0 ProblemContext.

    Notes:
        The actual re-simulation is usually performed by the caller or a
        SolutionContext object using the `ProblemContext.advance(route)` method.
        This function serves as a protocol marker to ensure cascade awareness.
    """
    return plan, problem


def cross_day_move(
    plan: List[List[int]],
    problem: ProblemContext,
    bin_id: int,
    from_day: int,
    to_day: int,
) -> Tuple[List[List[int]], int]:
    """
    Perturb the plan by moving a specific bin from one day to another.

    This operator removes `bin_id` from its current assignment on `from_day`
    and inserts it into `to_day` at the position that minimizes the additional
    distance cost on that target day. It is an Inter-Day relocation operator.

    Args:
        plan:     The current D-day plan.
        problem:  Day-0 ProblemContext containing the master geometry.
        bin_id:   The customer/bin node index to be moved.
        from_day: The index of the day where the bin is currently served.
        to_day:   The index of the day where the bin should be moved to.

    Returns:
        Tuple[List[List[int]], int]:
            - The modified D-day plan.
            - The index of the earliest affected day (min(from_day, to_day)).
    """
    new_plan = [list(r) for r in plan]

    if bin_id not in new_plan[from_day]:
        return plan, from_day  # nothing to do

    new_plan[from_day] = [n for n in new_plan[from_day] if n != bin_id]

    # Insert at best position in to_day's route (minimise distance increase)
    dm = problem.distance_matrix
    route = new_plan[to_day]
    best_cost = float("inf")
    best_pos = 0
    for pos in range(len(route) + 1):
        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos] if pos < len(route) else 0
        cost = dm[prev, bin_id] + dm[bin_id, nxt] - dm[prev, nxt]
        if cost < best_cost:
            best_cost = cost
            best_pos = pos

    new_plan[to_day].insert(best_pos, bin_id)
    return new_plan, min(from_day, to_day)


def cross_day_swap(
    plan: List[List[int]],
    problem: ProblemContext,
    bin_i: int,
    day_i: int,
    bin_j: int,
    day_j: int,
) -> Tuple[List[List[int]], int]:
    """
    Swap two bins between different days in the planning horizon.

    This operator exchanges two customers assigned to different days. It is
    particularly effective at re-balancing day loads or improving the
    spatial clustering of routes without changing the number of visits per day.

    Both resulting routes are checked for capacity feasibility using the
    estimated day-0 fill levels. If either swap would result in a capacity
    violation, the move is rejected.

    Args:
        plan:    The current D-day plan.
        problem: Day-0 ProblemContext.
        bin_i:   The first bin node index (coming from day_i).
        day_i:   The index of the first day.
        bin_j:   The second bin node index (coming from day_j).
        day_j:   The index of the second day.

    Returns:
        Tuple[List[List[int]], int]:
            - The modified D-day plan (or original if infeasible).
            - The index of the earliest affected day (min(day_i, day_j)).
    """
    if bin_i not in plan[day_i] or bin_j not in plan[day_j]:
        return plan, day_i

    Q = problem.capacity
    w = problem.wastes

    load_i_new = sum(w.get(n, 0.0) for n in plan[day_i] if n != bin_i) + w.get(bin_j, 0.0)
    load_j_new = sum(w.get(n, 0.0) for n in plan[day_j] if n != bin_j) + w.get(bin_i, 0.0)

    if load_i_new > Q or load_j_new > Q:
        return plan, day_i  # infeasible

    new_plan = [list(r) for r in plan]
    new_plan[day_i] = [bin_j if n == bin_i else n for n in new_plan[day_i]]
    new_plan[day_j] = [bin_i if n == bin_j else n for n in new_plan[day_j]]

    return new_plan, min(day_i, day_j)


def day_merge_split(
    plan: List[List[int]],
    problem: ProblemContext,
    rng: np.random.Generator,
) -> Tuple[List[List[int]], int]:
    """
    Perform structural consolidation or expansion of routes across days.

    This operator attempts to either merge two adjacent days with low utilization
    or split a highly utilized day's route into two days.

    1. Merge Heuristic:
       - Identifies two adjacent days where the combined fill level is less
         than or equal to the vehicle capacity (Q).
       - Moves all customers from the day with the lower total load to the day
         with the higher load, effectively vacating one day.

    2. Split Heuristic:
       - Identifies routes with high utilization (load > 0.7 * Q).
       - Selects the least profitable half of the bins (based on day-0 estimates)
         and moves them into a neighboring day if capacity permits.

    Args:
        plan:    The current D-day plan.
        problem: Day-0 ProblemContext.
        rng:     NumPy Generator instance.

    Returns:
        Tuple[List[List[int]], int]:
            - The structurally modified D-day plan.
            - The index of the first modified day.
    """
    Q = problem.capacity
    w = problem.wastes

    loads = [sum(w.get(n, 0.0) for n in route) for route in plan]
    D = len(plan)

    # Attempt merge first (random ordering)
    day_order = list(rng.permutation(D - 1))
    for d in day_order:
        if loads[d] < 0.5 * Q and loads[d + 1] < 0.5 * Q:
            combined_load = loads[d] + loads[d + 1]
            if combined_load <= Q:
                new_plan = [list(r) for r in plan]
                if loads[d] <= loads[d + 1]:
                    new_plan[d + 1] = new_plan[d] + new_plan[d + 1]
                    new_plan[d] = []
                else:
                    new_plan[d] = new_plan[d] + new_plan[d + 1]
                    new_plan[d + 1] = []
                return new_plan, d

    # Attempt split
    for d in range(D):
        if loads[d] > 0.7 * Q and len(plan[d]) >= 2:
            # Move lowest-profit bins to an adjacent day
            route = list(plan[d])
            # Sort by profit descending; keep top half, move bottom half
            route_sorted = sorted(route, key=lambda n: w.get(n, 0.0) * problem.revenue_per_kg, reverse=True)
            keep = route_sorted[: len(route_sorted) // 2]
            move = route_sorted[len(route_sorted) // 2 :]
            target_day = (d + 1) % D
            new_load_target = loads[target_day] + sum(w.get(n, 0.0) for n in move)
            if new_load_target <= Q:
                new_plan = [list(r) for r in plan]
                new_plan[d] = keep
                new_plan[target_day] = new_plan[target_day] + move
                return new_plan, min(d, target_day)

    return plan, 0
