"""
BCP (Integer Programming) engine using Gurobi for Build Optimization.
"""

from typing import Any, Dict, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None

from core.problem import BuildProblem


def run_bcp_gurobi(
    problem: BuildProblem,
    budget: float,
    values: Dict[str, Any],
    **kwargs: Any,
) -> Tuple[np.ndarray, float]:
    """
    Solve the Build Optimization problem (MCKP) using Gurobi.
    """
    if gp is None:
        return problem.greedy_solution(), 0.0

    # 1. Create model
    env = kwargs.get("env")
    model = gp.Model("BuildOptimization", env=env)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", values.get("time_limit", 30))

    # 2. Variables
    x = {}
    item_scores = (
        (problem.stat_matrix @ problem.stat_weights) + problem.rarities * problem.rarity_bonus + problem.slot_bonus
    )

    for slot_idx in range(problem.num_slots):
        item_indices = np.where(problem.slot_ids == slot_idx)[0]
        for i_idx in item_indices:
            x[slot_idx, i_idx] = model.addVar(vtype=GRB.BINARY, name=f"x_{slot_idx}_{i_idx}")

    # 3. Constraints
    for slot_idx in range(problem.num_slots):
        item_indices = np.where(problem.slot_ids == slot_idx)[0]
        if len(item_indices) > 0:
            model.addConstr(gp.quicksum(x[slot_idx, i_idx] for i_idx in item_indices) <= 1)

    model.addConstr(gp.quicksum(problem.costs[i_idx] * x[slot_idx, i_idx] for (slot_idx, i_idx) in x.keys()) <= budget)

    # 4. Objective
    model.setObjective(
        gp.quicksum(item_scores[i_idx] * x[slot_idx, i_idx] for (slot_idx, i_idx) in x.keys()), GRB.MAXIMIZE
    )

    # 5. Optimize
    model.optimize()

    # 6. Extract result
    best_build = np.full(problem.num_slots, -1, dtype=np.int64)
    if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        for (slot_idx, i_idx), var in x.items():
            if var.X > 0.5:
                best_build[slot_idx] = i_idx
        return best_build, problem.evaluate(best_build)

    return problem.greedy_solution(), 0.0
