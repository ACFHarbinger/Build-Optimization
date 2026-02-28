"""
BCP (Integer Programming) engine using OR-Tools for Build Optimization.
"""

from typing import Any, Dict, Tuple

import numpy as np
from ortools.linear_solver import pywraplp

from core.problem import BuildProblem


def run_bcp_ortools(
    problem: BuildProblem,
    budget: float,
    values: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[np.ndarray, float]:
    """
    Solve the Build Optimization problem (MCKP) using OR-Tools.
    """
    # 1. Create solver
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return problem.greedy_solution(), 0.0

    # 2. Variables: x[slot, item]
    x = {}
    item_scores = (
        (problem.stat_matrix @ problem.stat_weights) + problem.rarities * problem.rarity_bonus + problem.slot_bonus
    )

    for slot_idx in range(problem.num_slots):
        item_indices = np.where(problem.slot_ids == slot_idx)[0]
        for i_idx in item_indices:
            x[slot_idx, i_idx] = solver.BoolVar(f"x_{slot_idx}_{i_idx}")

    # 3. Constraints
    # One item per slot
    for slot_idx in range(problem.num_slots):
        item_indices = np.where(problem.slot_ids == slot_idx)[0]
        if len(item_indices) > 0:
            solver.Add(solver.Sum([x[slot_idx, i_idx] for i_idx in item_indices]) <= 1)

    # Budget constraint
    solver.Add(solver.Sum([problem.costs[i_idx] * x[slot_idx, i_idx] for (slot_idx, i_idx) in x.keys()]) <= budget)

    # 4. Objective
    objective = solver.Objective()
    for (_slot_idx, i_idx), var in x.items():
        objective.SetCoefficient(var, float(item_scores[i_idx]))
    objective.SetMaximization()

    # 5. Solve
    time_limit = values.get("time_limit", 30)
    solver.set_time_limit(int(time_limit * 1000))
    status = solver.Solve()

    # 6. Extract result
    best_build = np.full(problem.num_slots, -1, dtype=np.int64)
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        for (slot_idx, i_idx), var in x.items():
            if var.solution_value() > 0.5:
                best_build[slot_idx] = i_idx
        return best_build, problem.evaluate(best_build)

    return problem.greedy_solution(), 0.0
