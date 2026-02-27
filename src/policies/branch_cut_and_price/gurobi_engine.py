"""
Gurobi engine for Branch-Cut-and-Price module.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _setup_model(values: Dict[str, Any], env: Optional[gp.Env]) -> gp.Model:
    """Initialize Gurobi model with parameters."""
    model = gp.Model("CVRP", env=env) if env else gp.Model("CVRP")
    model.setParam("TimeLimit", values.get("time_limit", 30))
    model.setParam("MIPGap", 0.05)
    return model


def _create_variables(
    model: gp.Model,
    nodes: List[int],
    customers: List[int],
    wastes: Dict[int, float],
    capacity: float,
) -> Tuple[Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var], Dict[int, gp.Var]]:
    """Create decision variables x, y, u."""
    # x[i,j]: 1 if edge (i,j) used.
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # y[i]: 1 if node i is visited (Waste Collecting)
    y = {}
    for i in customers:
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    # u[i]: load after visiting node i (MTZ)
    u = {}
    for i in customers:
        u[i] = model.addVar(lb=wastes.get(i, 0), ub=capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    return x, y, u


def _create_objective(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    dist_matrix: Any,
    wastes: Dict[int, float],
    nodes: List[int],
    customers: List[int],
    R: float,
    C: float,
    mandatory_nodes: Optional[Set[int]],
) -> None:
    """Set the objective function: Minimize Cost + Penalties."""
    travel_cost = gp.quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)

    revenue_penalty = gp.LinExpr(0)
    m_nodes = mandatory_nodes if mandatory_nodes is not None else set()
    for i in customers:
        d = wastes.get(i, 0)
        rev = d * R
        if i in m_nodes:
            # Must Visit constraint
            model.addConstr(y[i] == 1, name=f"must_visit_{i}")
        else:
            # Penalty if y[i] is 0 -> (1 - y[i]) * rev
            revenue_penalty += (1 - y[i]) * rev

    model.setObjective(travel_cost + revenue_penalty, GRB.MINIMIZE)


def _add_constraints(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    u: Dict[int, gp.Var],
    nodes: List[int],
    customers: List[int],
    wastes: Dict[int, float],
    capacity: float,
) -> None:
    """Add flow conservation and capacity constraints."""
    # Flow Conservation
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"in_{i}")

    # MTZ Constraints (Subtour elimination and load tracking)
    for i in customers:
        for j in customers:
            if i != j:
                d_j = wastes.get(j, 0)
                model.addConstr(u[j] >= u[i] + d_j - capacity * (1 - x[i, j]), name=f"mtz_{i}_{j}")


def _extract_solution(
    model: gp.Model, x: Dict[Tuple[int, int], gp.Var], nodes: List[int]
) -> Tuple[List[List[int]], float]:
    """Extract optimal routes from the model."""
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or model.SolCount == 0:
        return [], 0.0

    active_edges = [edge for edge, var in x.items() if var.X > 0.5]
    adj: Dict[int, List[int]] = {i: [] for i in nodes}
    for i, j in active_edges:
        adj[i].append(j)

    routes = []
    # Find all paths starting from depot (0)
    for start_node in adj[0]:
        route = [start_node]
        current = start_node
        while current != 0 and current in adj and adj[current]:
            next_node = adj[current][0]
            if next_node != 0:
                route.append(next_node)
            current = next_node
        routes.append(route)

    return routes, model.objVal


def run_bcp_gurobi(
    dist_matrix: Any,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve Waste-Collecting CVRP using Gurobi MIP solver.

    Args:
        dist_matrix: NxN distance matrix.
        wastes: Dictionary of node wastes.
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Config dictionary (time_limit).
        mandatory_nodes: List of mandatory node indices.
        env: Gurobi environment (Optional).

    Returns:
        Tuple[List[List[int]], float]: (routes, total_cost)
    """
    m_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Identifying Customer Nodes: Indices 1..N
    num_nodes = len(dist_matrix)
    nodes = list(range(num_nodes))
    customers = list(range(1, num_nodes))

    model = _setup_model(values, env)
    x, y, u = _create_variables(model, nodes, customers, wastes, capacity)
    _create_objective(model, x, y, dist_matrix, wastes, nodes, customers, R, C, m_set)
    _add_constraints(model, x, y, u, nodes, customers, wastes, capacity)

    model.optimize()

    if recorder is not None:
        solved = int(model.SolCount > 0)
        recorder.record(
            engine="gurobi",
            solved=solved,
            obj_val=model.ObjVal if solved else 0.0,
            mip_gap=model.MIPGap if solved else 1.0,
            obj_bound=model.ObjBound if solved else 0.0,
        )

    return _extract_solution(model, x, nodes)
