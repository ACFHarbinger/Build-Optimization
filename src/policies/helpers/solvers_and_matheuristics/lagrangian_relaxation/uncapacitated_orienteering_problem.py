"""
Uncapacitated Orienteering Problem (OP) inner solver for Lagrangian Relaxation.

When the capacity constraint Σ_{i∈A_d} w_i ≤ Q is relaxed with multiplier λ ≥ 0,
the modified revenue per customer i becomes:

    p̃_i(λ) = (r_w - λ) · w_i

The resulting subproblem selects a subset A ⊆ customers and a tour that
maximises total modified revenue minus travel cost, with NO capacity constraint:

    max  Σ_{i∈A} p̃_i(λ) · y_i  -  C · dist(A)
    s.t. flow conservation, subtour elimination (DFJ lazy cuts), y_i ∈ {0,1}

Customers with p̃_i(λ) ≤ 0 (and not in forced_in) are excluded a-priori to keep
the Gurobi model small. The full Lagrangian bound is then:

    L(λ) = OP_objective(λ) + λ · Q

References:
    Dantzig, G., Fulkerson, R., & Johnson, S. (1954). "Solution of a large-scale
    traveling-salesman problem". Journal of Operations Research, 2(4), 393-410.
"""

from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _op_dfj_callback(model: gp.Model, where: int) -> None:
    """
    DFJ lazy subtour elimination callback for the uncapacitated OP.

    Triggered at integer solutions (MIPSOL). Detects connected components that
    do not contain the depot (node 0) and injects a cut forcing at least one
    outgoing edge from each subtour component.
    """
    if where != GRB.Callback.MIPSOL:
        return

    x_vars: Dict[Tuple[int, int], gp.Var] = model._x_vars
    num_nodes: int = model._num_nodes

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for (i, j), var in x_vars.items():
        if model.cbGetSolution(var) > 0.5:
            G.add_edge(i, j)

    for component in nx.connected_components(G):
        if 0 not in component and len(component) >= 2:
            rest = set(range(num_nodes)) - component
            outgoing = [x_vars[i, j] for i in component for j in rest if (i, j) in x_vars]
            if outgoing:
                model.cbLazy(quicksum(outgoing) >= 1)


def solve_uncapacitated_op(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    lam: float,
    R: float,
    C: float,
    forced_in: Optional[Set[int]] = None,
    forced_out: Optional[Set[int]] = None,
    time_limit: float = 10.0,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[Set[int], float, float]:
    """
    Solve the uncapacitated Orienteering Problem (OP) at a given λ.

    The modified profit for customer i is p̃_i(λ) = (R - λ) · w_i. Customers with
    p̃_i(λ) ≤ 0 (and not in forced_in) are excluded a-priori. The problem is built
    over the reduced eligible node set and solved with Gurobi using DFJ lazy cuts.

    Args:
        dist_matrix: Symmetric distance matrix (n × n), index 0 = depot.
        wastes: {local_customer_id → fill_level} for nodes 1..n-1.
        lam: Current Lagrange multiplier λ ≥ 0.
        R: Revenue coefficient per unit waste (r_w).
        C: Cost coefficient per unit distance (c_km).
        forced_in: Customer indices that MUST be visited (e.g. mandatory or B&B fixes).
        forced_out: Customer indices that MUST NOT be visited (B&B fixes).
        time_limit: Gurobi time limit for this solve (seconds).
        seed: Gurobi random seed for reproducibility.
        env: Optional shared Gurobi environment.
        recorder: Optional telemetry recorder.

    Returns:
        (visited_set, op_objective, dist_cost) where:
            visited_set  – global customer indices selected by the OP.
            op_objective – Σ p̃_i·y_i - C·dist  (without the +λQ term).
            dist_cost    – raw distance traveled.
    """
    forced_in = forced_in or set()
    forced_out = forced_out or set()

    customers = list(range(1, len(dist_matrix)))

    # A-priori elimination: skip unprofitable free customers
    eligible: List[int] = []
    for i in customers:
        if i in forced_out:
            continue
        if i in forced_in or (R - lam) * wastes.get(i, 0.0) > 0.0:
            eligible.append(i)

    if not eligible:
        return set(), 0.0, 0.0

    # Build reduced problem over {depot=0} ∪ eligible
    active_nodes = [0] + eligible
    n = len(active_nodes)
    local_idx = {g: l for l, g in enumerate(active_nodes)}

    local_dm = np.zeros((n, n))
    for li, gi in enumerate(active_nodes):
        for lj, gj in enumerate(active_nodes):
            local_dm[li, lj] = dist_matrix[gi][gj]

    local_wastes = {local_idx[i]: wastes.get(i, 0.0) for i in eligible}
    local_forced_in = {local_idx[i] for i in forced_in if i in local_idx}
    local_customers = list(range(1, n))

    model = gp.Model("BB_LR_OP", env=env) if env else gp.Model("BB_LR_OP")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Seed", seed)
    model.Params.LazyConstraints = 1

    x: Dict[Tuple[int, int], gp.Var] = {}
    for li in range(n):
        for lj in range(n):
            if li != lj:
                x[li, lj] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x_{li}_{lj}")

    y: Dict[int, gp.Var] = {}
    for li in local_customers:
        lb = 1.0 if li in local_forced_in else 0.0
        y[li] = model.addVar(lb=lb, ub=1, vtype=GRB.BINARY, name=f"y_{li}")

    # Objective: max Σ p̃_i·y_i - C·dist
    revenue = quicksum((R - lam) * local_wastes.get(li, 0.0) * y[li] for li in local_customers)
    travel = quicksum(local_dm[li][lj] * C * x[li, lj] for (li, lj) in x)
    model.setObjective(revenue - travel, GRB.MAXIMIZE)

    # Flow conservation
    for li in local_customers:
        model.addConstr(quicksum(x[li, lj] for lj in range(n) if li != lj) == y[li], name=f"out_{li}")
        model.addConstr(quicksum(x[lj, li] for lj in range(n) if li != lj) == y[li], name=f"in_{li}")

    # Single vehicle: at most one route from depot
    model.addConstr(quicksum(x[0, lj] for lj in local_customers) <= 1, name="depot_out")
    model.addConstr(
        quicksum(x[lj, 0] for lj in local_customers) == quicksum(x[0, lj] for lj in local_customers),
        name="depot_balance",
    )

    model._x_vars = x
    model._num_nodes = n

    model.optimize(_op_dfj_callback)

    if model.SolCount == 0:
        return set(forced_in), 0.0, 0.0

    visited_local = {li for li, var in y.items() if var.X > 0.5}
    visited_global = {active_nodes[li] for li in visited_local}
    dist_cost = sum(local_dm[li][lj] for (li, lj), var in x.items() if var.X > 0.5)
    op_obj = float(model.ObjVal)

    if recorder:
        recorder.record(engine="bb_lr_op", lam=lam, op_obj=op_obj, n_visited=len(visited_global))

    return visited_global, op_obj, dist_cost
