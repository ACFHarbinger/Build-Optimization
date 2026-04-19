"""
Relaxation Enforced Neighborhood Search (RENS) solver.
"""

from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _setup_rens_model(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    seed: int,
    env: Optional[gp.Env],
) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
    """
    Build the base VRP model for RENS with continuous variables.

    This model serves as the LP relaxation to identify integer-valued
    features of the optimal fractional solution.

    Args:
        dist_matrix: Pairwise node distances.
        wastes: Demand/waste levels at each node.
        capacity: Maximum vehicle load.
        R: Revenue multiplier for collected waste.
        C: Cost multiplier for distance traveled.
        mandatory_nodes: Nodes that must be visited.
        seed: Random seed for Gurobi parameters.
        env: Optional Gurobi environment for shared settings.

    Returns:
        A tuple of (model, x_vars, y_vars).
    """
    num_nodes = len(dist_matrix)
    nodes = list(range(num_nodes))
    customers = list(range(1, num_nodes))
    m_set = set(mandatory_nodes)

    model = gp.Model("RENS_VRPP", env=env) if env else gp.Model("RENS_VRPP")
    model.setParam("OutputFlag", 0)  # Suppress internal Gurobi logging
    model.setParam("Seed", seed)

    # 1. VARIABLE DECLARATION (Continuous [0, 1] for LP Relaxation)
    # x[i,j]: 1 if vehicle travels from node i to j.
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    # y[i]: 1 if node i is visited by the vehicle.
    y = {}
    for i in customers:
        y[i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{i}")

    # u[i]: Load level after visiting node i (subtour elimination / capacity).
    u = {}
    for i in customers:
        u[i] = model.addVar(lb=wastes.get(i, 0), ub=capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    # 2. OBJECTIVE FUNCTION: Maximize Profit = Revenue - Travel Cost
    travel_cost = quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)
    revenue = quicksum(wastes.get(i, 0) * R * y[i] for i in customers)
    model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

    # 3. CONSTRAINTS
    # Degree constraints for each customer
    for i in customers:
        model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"out_{i}")
        model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"in_{i}")

    # Depot usage (at most one route starting from depot)
    model.addConstr(quicksum(x[0, j] for j in customers) <= 1, name="depot_out")
    model.addConstr(quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers), name="depot_balance")

    # MTZ Constraints: Subtour elimination and node-wise capacity tracking
    for i in customers:
        for j in customers:
            if i != j:
                dj = wastes.get(j, 0)
                # If x[i,j]=1, then u[j] >= u[i] + dj
                model.addConstr(u[j] >= u[i] + dj - capacity * (1 - x[i, j]), name=f"mtz_{i}_{j}")

    # Mandatory service requirements
    for i in m_set:
        model.addConstr(y[i] == 1, name=f"mandatory_{i}")

    return model, x, y


def _apply_restrictions(model: gp.Model, x: Dict[Tuple[int, int], gp.Var], y: Dict[int, gp.Var]) -> None:
    """
    Apply RENS rounding restrictions to the model variables.

    Reference: Berthold (2009)
    The neighborhood is defined by fixing all variables that are
    integer-valued in the LP solution to those values, and
    restricting the remaining variables to be binary.
    """
    # 1. Collect fractional variables for potential neighborhood expansion if needed
    fractional = []

    for var in x.values():
        val = var.X
        if val < 1e-6:
            var.LB, var.UB = 0, 0
        elif val > 1.0 - 1e-6:
            var.LB, var.UB = 1, 1
        else:
            fractional.append(var)
        var.VType = GRB.BINARY

    for var in y.values():
        val = var.X
        if val < 1e-6:
            var.LB, var.UB = 0, 0
        elif val > 1.0 - 1e-6:
            var.LB, var.UB = 1, 1
        else:
            fractional.append(var)
        var.VType = GRB.BINARY

    # RENS property: if no fractional variables, sub-MIP is trivial


def run_rens_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    time_limit: float = 60.0,
    lp_time_limit: float = 10.0,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve VRPP using Relaxation Enforced Neighborhood Search (RENS).

    RENS strategy:
    1. Build LP relaxation (all binary variables treated as continuous [0,1]).
    2. Solve LP to find fractional optimal values.
    3. Fix variables with integer values in LP solution.
    4. Solve a restricted MIP on the remaining binary variables.

    Returns:
        Tuple of (full_tour, objective_profit_value, primary_travel_cost).
    """
    # Phase 1: Setup and Relaxation Solve
    model, x, y = _setup_rens_model(dist_matrix, wastes, capacity, R, C, mandatory_nodes, seed, env)

    model.setParam("TimeLimit", lp_time_limit)
    model.optimize()

    # Check if a relaxation solution was found
    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    # Phase 2: Variable Fixing and Sub-MIP Solve
    _apply_restrictions(model, x, y)

    # Check if any binary variables are still unfixed (fractional in LP)
    unfixed = [v for v in model.getVars() if v.VType == GRB.BINARY and v.LB < v.UB]

    if not unfixed:
        # LP solution was already integer! No need for sub-MIP.
        pass
    else:
        rem_time = max(1.0, time_limit - model.Runtime)
        model.setParam("TimeLimit", rem_time)
        model.setParam("MIPGap", mip_gap)
        # Focus heuristic search as RENS is primarily a primal start heuristic
        model.setParam("MIPFocus", 1)
        # NOTE: _dfj_subtour_elimination_callback is not defined in this file.
        # If this callback is intended to be used, its definition must be added.
        # For now, calling without callback to maintain syntactic correctness.
        model.optimize()

    # Extract results
    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    # Adjacency map for tour reconstruction
    active_edges = [edge for edge, var in x.items() if var.X > 0.5]
    adj: Dict[int, List[int]] = {i: [] for i in range(len(dist_matrix))}
    for i, j in active_edges:
        adj[i].append(j)

    full_tour, current = [0], 0
    while adj.get(current):
        nx_node = adj[current].pop(0)
        full_tour.append(nx_node)
        current = nx_node
        if current == 0:
            break

    # Close the loop if needed (standard VRP)
    if full_tour[-1] != 0:
        full_tour.append(0)

    obj, cost = model.ObjVal, sum(dist_matrix[i][j] for i, j in active_edges)

    # Metrics recording
    if recorder:
        recorder.record(engine="rens", solved=1, obj_val=obj, cost=cost)

    return full_tour, float(obj), float(cost)
